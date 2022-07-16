#include <nncase/debug.h>

namespace fs = std::filesystem;

int a = 1;
bool append = false;
std::string currentOp = "";
fs::path dump_root = "";

static int get_a() { return a; }
static void incr_a() {
    a++;
    append = false;
}

void set_dump_root(std::string root) {
    // todo:maybe should change path in pytest
    dump_root.clear();
    fs::path p(root);
    dump_root = (p / "Runtime");
}

fs::path dump_path() {
    auto p = dump_root / (std::to_string(get_a()) + currentOp);
    if (!fs::exists(dump_root) && dump_root != "") {
        fs::create_directory(dump_root);
    }
    return p;
}

std::ofstream get_stream(const fs::path &path) {
    return append ? std::ofstream(path, std::ios_base::app)
                  : std::ofstream(path);
}

std::string to_str(const nncase::dims_t &shape) {
    std::stringstream stream;
    if (shape.size() == 0) {
        stream << "scalar\n";
    } else {
        for (auto d : shape) {
            stream << std::to_string(d) << " ";
        }
        stream << std::endl;
    }
    return stream.str();
}

void write_shape(const nncase::dims_t &shape) {
    auto path = fs::path(dump_root) / "9999shape";
    auto f = fs::exists(path) ? std::ofstream(path, std::ios::app)
                              : std::ofstream(path);
    f << currentOp << " :" << to_str(shape);
    f.close();
}

void dump_output_impl(nncase::value_t value, const fs::path &path, bool incr) {
    dump(
        value,
        [incr](auto &stream, auto &&value_tensor) {
            auto *data = value_tensor->to_host()
                             .unwrap()
                             ->buffer()
                             .as_host()
                             .unwrap()
                             .map(nncase::runtime::map_read)
                             .unwrap()
                             .buffer()
                             .data();
            if (incr) {
                write_shape(value_tensor->shape());
            }
#define RETURN_RESULT(_in_type)                                                \
    if (nncase::runtime::cmp_type<_in_type>(value_tensor->dtype())) {          \
        dump_data(stream, IN_CAST(_in_type, data), value_tensor);              \
    }
            RETURN_RESULT(bool);
            RETURN_RESULT(int32_t);
            RETURN_RESULT(uint32_t);
            RETURN_RESULT(int64_t);
            RETURN_RESULT(uint64_t);
            RETURN_RESULT(float);
        },
        path);
    if (incr) {
        incr_a();
    }
}

void dump_output(NNCASE_UNUSED nncase::value_t value) {
    dump_output_impl(value, dump_path(), true);
}

void dump_input(NNCASE_UNUSED nncase::value_t value,
                NNCASE_UNUSED std::string name) {
    dump_output_impl(value, fs::path(dump_path().string() + name), false);
}