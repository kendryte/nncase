/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "template.h"
#include <fmt/format.h>
#include <inja/inja.hpp>
#include <libzippp/libzippp.h>
#include <shaderc/shaderc.hpp>
#include <span>
#include <system_error>
#include <unordered_map>

using namespace libzippp;
using namespace nncase;
using namespace nncase::codegen::vulkan;

#if _MSC_VER
#include "resource.h"
#include <Windows.h>

#define THROW_WIN32_IF_NOT(x, fmt_str, ...)                                    \
    if (!(x)) {                                                                \
        auto err_code = GetLastError();                                        \
        auto err_msg = std::system_category().message(err_code);               \
        throw std::system_error(err_code, std::system_category(),              \
                                fmt::format(fmt_str, err_msg, __VA_ARGS__));   \
    }

extern HMODULE g_vulkan_module_handle;

namespace {
struct xz_res {
    std::span<const uint8_t> data;

    xz_res() {
        auto hres =
            FindResourceW(g_vulkan_module_handle,
                          MAKEINTRESOURCEW(IDR_VULKAN_TEMPLATES), L"Binary");
        THROW_WIN32_IF_NOT(hres, "Cannot find resource: {}",
                           "Vulkan Templates");
        auto size = SizeofResource(g_vulkan_module_handle, hres);
        auto hmem = LoadResource(g_vulkan_module_handle, hres);
        THROW_WIN32_IF_NOT(hmem, "Cannot load resource: {}",
                           "Vulkan Templates");
        auto res_data = LockResource(hmem);
        data = {reinterpret_cast<const uint8_t *>(res_data), (size_t)size};
    }
};
} // namespace
#else
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <nncase/runtime/incbin.h>
INCBIN(templates_xz, "vulkan_templates_xz.xz");

namespace {
struct xz_res {
    std::span<const uint8_t> data;

    xz_res() {
        data = {reinterpret_cast<const uint8_t *>(gtemplates_xz_data),
                (size_t)gtemplates_xz_size};
    }
};
} // namespace
#endif

namespace {
class xz_reader {
  public:
    xz_reader()
        : archive_(ZipArchive::fromBuffer(xz_res_.data.data(),
                                          (libzippp_uint32)xz_res_.data.size(),
                                          ZipArchive::ReadOnly, true)) {
        if (!archive_)
            throw std::runtime_error("Load vulkan templates archive failed");
    }

    std::string read(const std::string &name) {
        auto entry = archive_->getEntry(name, true, true);
        if (entry.isNull())
            throw std::runtime_error("Vulkan template not found: " + name);
        return entry.readAsText();
    }

  private:
    xz_res xz_res_;
    std::unique_ptr<ZipArchive> archive_;
};

class compiler {
  public:
    compiler() {
        shader_options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                             shaderc_env_version_vulkan_1_1);
        shader_options_.SetTargetSpirv(shaderc_spirv_version_1_3);
    }

    std::string render(const std::string &template_name,
                       const nlohmann::json &context) {
        auto templ = load_template(template_name);
        return env_.render(templ, context);
    }

    std::vector<uint32_t> render_and_compile(const std::string &template_name,
                                             const compile_options &options) {
        auto is_hlsl = template_name.ends_with(".hlsl");
        shader_options_.SetSourceLanguage(is_hlsl
                                              ? shaderc_source_language_hlsl
                                              : shaderc_source_language_glsl);
        auto source = render(template_name, options.context);

        if (options.dump_asm) {
            {
                std::ofstream hlsl_file(
                    options.dump_dir /
                        (options.function_name + (is_hlsl ? ".hlsl" : ".comp")),
                    std::ios::out | std::ios::binary);
                hlsl_file << source;
            }

            {
                auto shader_asm = shader_compiler_.CompileGlslToSpvAssembly(
                    source, shaderc_compute_shader, template_name.c_str(),
                    shader_options_);
                if (shader_asm.GetCompilationStatus() !=
                    shaderc_compilation_status_success)
                    throw std::runtime_error(shader_asm.GetErrorMessage());

                std::ofstream asm_file(options.dump_dir /
                                           (options.function_name + ".spv"),
                                       std::ios::out | std::ios::binary);
                asm_file << std::string(shader_asm.begin(), shader_asm.end());
            }
        }

        auto shader_spv = shader_compiler_.CompileGlslToSpv(
            source, shaderc_compute_shader, template_name.c_str(),
            shader_options_);
        if (shader_spv.GetCompilationStatus() !=
            shaderc_compilation_status_success)
            throw std::runtime_error(shader_spv.GetErrorMessage());
        return {shader_spv.begin(), shader_spv.end()};
    }

  private:
    inja::Template &load_template(const std::string &template_name) {
        auto it = template_cache_.find(template_name);
        if (it == template_cache_.end())
            it = template_cache_
                     .emplace(template_name,
                              env_.parse(reader_.read(template_name)))
                     .first;
        return it->second;
    }

  private:
    xz_reader reader_;
    inja::Environment env_;
    std::unordered_map<std::string, inja::Template> template_cache_;
    shaderc::Compiler shader_compiler_;
    shaderc::CompileOptions shader_options_;
};
} // namespace

std::vector<uint32_t>
codegen::vulkan::render_and_compile(const std::string &template_name,
                                    const compile_options &options) {
    static compiler cp;
    return cp.render_and_compile(template_name, options);
}
