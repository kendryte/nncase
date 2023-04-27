from distutils.command.install_data import install_data
import imp
from posixpath import dirname
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
import shutil
import os
import platform
import sys
import io
import re
import time

# See ref: https://stackoverflow.com/a/51575996


class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info

    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        self.outfiles = self.distribution.data_files


class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True

        bin_dir = self.distribution.bin_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = [os.path.join(root, _lib) for root, _, files in
                os.walk(bin_dir) for _lib in files if
                os.path.isfile(os.path.join(root, _lib)) and
                os.path.splitext(_lib)[-1] in [".dll", ".so", ".dylib", ".json"]
                and not (_lib.startswith("python") or _lib.startswith("_nncase"))]

        for lib in libs:
            shutil.move(lib, os.path.join(self.build_dir,
                                          os.path.basename(lib)))

        # Mark the libs for installation, adding them to
        # distribution.data_files seems to ensure that setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list,
        # but that never seemed to add them to the installed-files.txt in the
        # egg-info, and the online recommendation seems to be adding libraries
        # into eager_resources in the call to setup(), which I think puts them
        # in data_files anyways.
        #
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [os.path.join(self.install_dir,
                                                     os.path.basename(lib))
                                        for lib in libs]

        # Must be forced to run after adding the libs to data_files

        self.distribution.run_command("install_data")

        super().run()


class InstallCMakeScripts(install_scripts):
    """
    Install the scripts in the build dir
    """

    def run(self):
        """
        Copy the required directory to the build directory and super().run()
        """

        self.announce("Moving scripts files", level=3)

        # Scripts were already built in a previous step

        self.skip_build = True

        bin_dir = self.distribution.bin_dir

        scripts_dirs = [os.path.join(bin_dir, _dir) for _dir in
                        os.listdir(bin_dir) if
                        os.path.isdir(os.path.join(bin_dir, _dir))]

        for scripts_dir in scripts_dirs:

            shutil.move(scripts_dir,
                        os.path.join(self.build_dir,
                                     os.path.basename(scripts_dir)))

        # Mark the scripts for installation, adding them to
        # distribution.scripts seems to ensure that the setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info

        self.distribution.scripts = scripts_dirs

        super().run()


class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext: Extension):
        """
        The steps required to build the extension
        """

        self.announce("Preparing the build environment", level=3)

        extpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        bin_dir = os.path.abspath(os.path.join(self.build_temp, 'install'))
        cmake_args = ['-G', 'Ninja']
        if platform.system() == 'Linux':
            cmake_args += ['-DCMAKE_C_COMPILER=gcc-10']
            cmake_args += ['-DCMAKE_CXX_COMPILER=g++-10']
        elif platform.system() == 'Windows':
            cmake_args += ['-DCMAKE_C_COMPILER=clang-cl']
            cmake_args += ['-DCMAKE_CXX_COMPILER=clang-cl']
        cmake_args += ['-DPython3_ROOT_DIR=' + os.path.dirname(sys.executable)]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        install_args = ['--prefix', bin_dir,
                        '--component', 'nncase-runtime']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Now that the necessary directories are created, build

        self.announce("Configuring cmake project", level=3)

        # Change your cmake arguments below as necessary
        # Below is just an example set of arguments for building Blender as a Python module

        self.spawn(['cmake', '-S' + ext.sourcedir, '-B' + self.build_temp] +
                   cmake_args)

        self.announce("Building binaries", level=3)

        self.spawn(["cmake", "--build", self.build_temp] + build_args)
        self.spawn(["cmake", "--install", self.build_temp] + install_args)

        # Build finished, now copy the files into the copy directory
        # The copy directory is the parent directory of the extension (.pyd)

        self.announce("Moving built python module", level=3)

        self.distribution.bin_dir = bin_dir

        pyd_path = [os.path.join(root, _pyd) for root, _, files in
                    os.walk(bin_dir) for _pyd in files if
                    os.path.isfile(os.path.join(root, _pyd)) and
                    os.path.splitext(_pyd)[0].startswith('_nncase') and
                    os.path.splitext(_pyd)[-1] in [".pyd", ".so"]][0]

        shutil.move(pyd_path, extpath)

        # copy nncase publish
        nncase_libs = [os.path.join(root, _lib) for root, _, files in
                os.walk(os.path.join(ext.sourcedir, 'install')) for _lib in files if
                os.path.isfile(os.path.join(root, _lib)) and
                os.path.splitext(_lib)[-1] in [".dll", ".so", ".dylib", ".json"]]

        for lib in nncase_libs:
            shutil.move(lib, os.path.join(bin_dir, 'nncase.deps', os.path.basename(lib)))

        # After build_ext is run, the following commands will run:
        #
        # install_lib
        # install_scripts
        #
        # These commands are subclassed above to avoid pitfalls that
        # setuptools tries to impose when installing these, as it usually
        # wants to build those libs and scripts as well or move them to a
        # different place. See comments above for additional information


def find_version():
    with io.open("CMakeLists.txt", encoding="utf8") as f:
        version_file = f.read()

    version_prefix = re.findall(r"NNCASE_VERSION \"(.+)\"", version_file)

    if version_prefix:
        version_suffix = time.strftime("%Y%m%d", time.localtime())
        return version_prefix[0] + "." + version_suffix
    raise RuntimeError("Unable to find version string.")


setup(name='nncase',
      version=find_version(),
      packages=['nncase'],
      package_dir={'': 'python'},
      ext_modules=[CMakeExtension(name="_nncase", sourcedir='.')],
      cmdclass={
          'build_ext': BuildCMakeExt,
          'install_data': InstallCMakeLibsData,
          'install_lib': InstallCMakeLibs
      }
      )
