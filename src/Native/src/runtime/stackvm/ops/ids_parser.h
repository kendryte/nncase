#pragma once
#include <map>

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

inline std::vector<std::string> split(const std::string &s, char delim = ' ') {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

inline std::string lookup([[maybe_unused]] std::string path, [[maybe_unused]] uint32_t module_id, [[maybe_unused]] uint32_t func_id)
{
//    return "";
    // todo: lookup once
    std::ifstream ids(path);
    std::stringstream buffer;
    buffer << ids.rdbuf();
    std::string ids_info(buffer.str());
    auto lines = split(ids_info, '\n');
    std::map<std::pair<int, int>, std::string> idMap;
    for (int i = 0; i < lines.size(); ++i) {
        auto info = split(lines[i], ' ');
        //        std::cout << lines[i] << std::endl;
        //        std::cout << info[0] << " " << info[1] << " " << info[2];
        idMap[std::pair(std::stoi(info[0]), std::stoi(info[1]))] = info[2];
    }
    auto fnName = idMap[std::pair(module_id, func_id)];
    if(fnName == "")
    {
        return std::to_string(module_id) + "_" + std::to_string(func_id);
    }
    return fnName;
    //    auto extcall_str = "extcall_" + std::to_string(module_id.as_u()) + "_"
    //    + std::to_string(func_id.as_u());
}