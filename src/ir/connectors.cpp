#include <ir/connectors.h>

using namespace nncase;
using namespace nncase::ir;

void input_connector::connect(output_connector &connector)
{
    if (type() != connector.type())
        throw std::runtime_error("Type must be same");

    if (!xt::same_shape(shape(), connector.shape()))
        throw std::runtime_error("Shapes must be same");

    if (connection_ != &connector)
    {
        clear_connection();
        connection_ = &connector;
        connector.connect(*this);
    }
}

void input_connector::clear_connection()
{
    auto from = connection_;
    if (from)
    {
        connection_ = nullptr;
        from->disconnect(*this);
    }
}

void output_connector::connect(input_connector &connector)
{
    if (std::find(std::begin(connections_), std::end(connections_), &connector) == std::end(connections_))
    {
        connector.connect(*this);
        connections_.emplace_back(&connector);
    }
}

void output_connector::disconnect(input_connector &connector)
{
    auto end = std::remove_if(connections_.begin(), connections_.end(), [&](auto conn) { return conn == &connector; });
    connections_.erase(end, connections_.end());
    connector.clear_connection();
}

void output_connector::clear_connections()
{
    std::vector<input_connector *> connections;
    connections_.swap(connections);
    for (auto conn : connections)
        conn->clear_connection();
}
