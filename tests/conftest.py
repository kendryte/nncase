def pytest_addoption(parser):
    parser.addoption(
        "--model_path",
        action="store",
        default=None,
        help="Path to the ONNX model file"
    )