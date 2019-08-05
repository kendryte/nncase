/* Copyright 2019 Canaan Inc.
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
#include <ir/connectors.h>

using namespace nncase;
using namespace nncase::ir;

void input_connector::connect(output_connector &connector)
{
    if (type() != connector.type())
        throw std::runtime_error("Type must be same");

    if (!xt::same_shape(shape(), connector.shape()))
        throw std::runtime_error("Shapes must be same, but got [" + to_string(shape()) + "] and [" + to_string(connector.shape()) + "]");

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
        connections_.emplace_back(&connector);
        connector.connect(*this);
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
