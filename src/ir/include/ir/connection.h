#pragma once

namespace nncase
{
namespace ir
{
    class input_connector;
    class output_connector;

    class connection
    {
    public:
        output_connector &from;
        input_connector &to;

        connection(output_connector &from, input_connector &to)
            : from(from), to(to)
        {
        }
    };
}
}
