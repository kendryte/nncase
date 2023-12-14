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
#include "pe_loader.h"
#include <Windows.h>
#include <nncase/runtime/result.h>

using namespace nncase::runtime;

#ifndef NDEBUG
#define THROW_WIN32_IF_NOT(x)                                                  \
    if (!(x)) {                                                                \
        throw std::system_error(GetLastError(), std::system_category());       \
    }
#else
// Protection flags for memory pages (Executable, Readable, Writeable)
static int ProtectionFlags[2][2][2] = {
    {
        // not executable
        {PAGE_NOACCESS, PAGE_WRITECOPY},
        {PAGE_READONLY, PAGE_READWRITE},
    },
    {
        // executable
        {PAGE_EXECUTE, PAGE_EXECUTE_WRITECOPY},
        {PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE},
    },
};
#endif

pe_loader::~pe_loader() {
    if (image_) {
#ifndef NDEBUG
        FreeModule((HMODULE)image_);
#else
        VirtualFree(image_, 0, MEM_RELEASE);
#endif
    }
}

void pe_loader::load(gsl::span<const gsl::byte> pe) {
#ifndef NDEBUG
    wchar_t temp_path[MAX_PATH];
    wchar_t temp_filename[MAX_PATH];

    THROW_WIN32_IF_NOT(GetTempPathW(std::size(temp_path), temp_path));
    THROW_WIN32_IF_NOT(
        GetTempFileNameW(temp_path, L"nncase.function.cpu.", 0, temp_filename));

    auto func_file =
        CreateFileW(temp_filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                    FILE_ATTRIBUTE_TEMPORARY, nullptr);
    THROW_WIN32_IF_NOT(func_file != INVALID_HANDLE_VALUE);
    THROW_WIN32_IF_NOT(
        WriteFile(func_file, pe.data(), pe.size_bytes(), nullptr, nullptr));
    CloseHandle(func_file);

    func_file = CreateFileW(
        temp_filename, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    THROW_WIN32_IF_NOT(func_file != INVALID_HANDLE_VALUE);

    auto func_mod = LoadLibraryW(temp_filename);
    THROW_WIN32_IF_NOT(func_mod);
    image_ = (gsl::byte *)func_mod;
#else
    auto dos_header = reinterpret_cast<const IMAGE_DOS_HEADER *>(pe.data());
    auto nt_header = reinterpret_cast<const IMAGE_NT_HEADERS *>(
        pe.data() + dos_header->e_lfanew);
    image_ = (gsl::byte *)VirtualAlloc(nullptr,
                                       nt_header->OptionalHeader.SizeOfImage,
                                       MEM_COMMIT, PAGE_READWRITE);

    // 1. copy header
    memcpy(image_, dos_header, nt_header->OptionalHeader.SizeOfHeaders);
    auto new_nt_header =
        reinterpret_cast<IMAGE_NT_HEADERS *>(image_ + dos_header->e_lfanew);
    new_nt_header->OptionalHeader.ImageBase = (ULONGLONG)image_;

    // 2. copy sections
    IMAGE_SECTION_HEADER *sections_base =
        reinterpret_cast<IMAGE_SECTION_HEADER *>(
            (size_t)new_nt_header + sizeof(DWORD) +
            (size_t)(sizeof(IMAGE_FILE_HEADER)) +
            (size_t)new_nt_header->FileHeader.SizeOfOptionalHeader);
    auto optional_section_size = new_nt_header->OptionalHeader.SectionAlignment;

    for (int i = 0; i < new_nt_header->FileHeader.NumberOfSections; i++) {
        auto &section = sections_base[i];
        size_t section_size;
        if (section.SizeOfRawData == 0) {
            // section doesn't contain data in the dll itself, but may define
            // uninitialized data
            section_size = optional_section_size;
            auto dest = image_ + section.VirtualAddress;
            memset(dest, 0, section_size);
            section.Misc.PhysicalAddress =
                (DWORD)((uintptr_t)dest & 0xffffffff);
        } else {
            section_size = section.SizeOfRawData;
            auto dest = image_ + section.VirtualAddress;
            memcpy(dest, pe.data() + section.PointerToRawData,
                   section.SizeOfRawData);
            section.Misc.PhysicalAddress =
                (DWORD)((uintptr_t)dest & 0xffffffff);
        }

        // determine protection flags based on characteristics
        auto executable =
            (section.Characteristics & IMAGE_SCN_MEM_EXECUTE) != 0;
        auto readable = (section.Characteristics & IMAGE_SCN_MEM_READ) != 0;
        auto writeable = (section.Characteristics & IMAGE_SCN_MEM_WRITE) != 0;
        auto protect = ProtectionFlags[executable][readable][writeable];
        if (section.Characteristics & IMAGE_SCN_MEM_NOT_CACHED) {
            protect |= PAGE_NOCACHE;
        }

        DWORD oldProtect;
        VirtualProtect(image_ + section.VirtualAddress, section_size, protect,
                       &oldProtect);
    }
#endif
}

void *pe_loader::entry() const noexcept {
    auto dos_header = reinterpret_cast<const IMAGE_DOS_HEADER *>(image_);
    auto nt_header = reinterpret_cast<const IMAGE_NT_HEADERS *>(
        image_ + dos_header->e_lfanew);
    return image_ + nt_header->OptionalHeader.AddressOfEntryPoint;
}
