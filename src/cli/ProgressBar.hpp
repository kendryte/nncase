#ifndef PROGRESSBAR_PROGRESSBAR_HPP
#define PROGRESSBAR_PROGRESSBAR_HPP

#include <chrono>
#include <iostream>

class progress_bar
{
private:
    size_t bar_width;
    const char complete_char = '=';
    const char incomplete_char = ' ';
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

public:
    progress_bar(size_t width, char complete, char incomplete)
        : bar_width { width }, complete_char { complete }, incomplete_char { incomplete } { }

    progress_bar(size_t width)
        : bar_width { width } { }

    void display(size_t ticks, size_t total_ticks) const
    {
        float progress = (float)ticks / total_ticks;
        size_t pos = (size_t)(bar_width * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        std::cout << "  [";

        for (size_t i = 0; i < bar_width; ++i)
        {
            if (i < pos)
                std::cout << complete_char;
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << incomplete_char;
        }
        std::cout << "] " << size_t(progress * 100.0) << "% "
                  << float(time_elapsed) / 1000.0 << "s\r";
        std::cout.flush();
    }

    void done() const
    {
        std::cout << std::endl;
    }
};

#endif //PROGRESSBAR_PROGRESSBAR_HPP
