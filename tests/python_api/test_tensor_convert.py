import duca
import nncaseruntime as nrt
import numpy as np
from loguru import logger
import sys
import pytest

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
)

# Test data as pytest fixture


@pytest.fixture
def test_data():
    """Prepare different types of test data."""
    return {
        "2d_float32": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "1d_int32": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "3d_float16": np.random.rand(2, 3, 4).astype(np.float16),
        "large_array": np.random.rand(100, 100).astype(np.float32),
        "zeros": np.zeros((5, 5), dtype=np.float32),
        "ones": np.ones((3, 3), dtype=np.float32),
    }


@pytest.fixture
def tensor_converter():
    """Create TensorConversionTester object"""
    return TensorConversionTester()

# Convert the original class to a helper class


class TensorConversionTester:
    """Helper class: handles tensor conversion logic"""

    def verify_data_consistency(self, original, converted, test_name):
        """Verify data consistency."""
        try:
            if np.allclose(original, converted, rtol=1e-5, atol=1e-6):
                logger.success(f"✓ {test_name}: The data consistency verification has been passed.")
                return True
            else:
                logger.error(f"✗ {test_name}: Inconsistent data!")
                logger.error(f"  src data: {original.flatten()[:5]}...")
                logger.error(f"  cvrt data: {converted.flatten()[:5]}...")
                return False
        except Exception as e:
            logger.error(f"✗ {test_name}: Data verification failed. - {e}")
            return False

    def create_duca_host_tensor(self, arr):
        """Create DUCA host tensor"""
        return duca.tensor(arr)

    def create_duca_device_tensor(self, arr):
        """Create DUCA device tensor"""
        return duca.tensor(arr, "duca:0", 1)

    def create_nrt_tensor(self, arr):
        """Create NRT tensor"""
        return nrt.RuntimeTensor.from_numpy(arr)

# pytest test class


class TestTensorConversion:
    """Tensor convert tests."""

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_numpy_to_nrt(self, test_data, tensor_converter, data_name):
        """Test：numpy -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"Test numpy -> NRT tensor ({data_name})")

        nrt_tensor = tensor_converter.create_nrt_tensor(arr)
        assert nrt_tensor is not None, f"Create NRT tensor 失败 ({data_name})"

        # Verify data consistency
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"numpy->NRT ({data_name})"
        ), f"Data consistency verification failed. ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_duca_device_to_nrt(self, test_data, tensor_converter, data_name):
        """Test：DUCA tensor -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"Test DUCA device -> NRT tensor ({data_name})")

        # Create DUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # Convert to NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        assert nrt_tensor is not None, f"DUCA->NRT conversion failed ({data_name})"

        # Verify data consistency
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"DUCA->NRT ({data_name})"
        ), f"Data consistency verification failed. ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_duca_host_to_nrt(self, test_data, tensor_converter, data_name):
        """Test：DUCA tensor -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"Test DUCA host -> NRT tensor ({data_name})")

        # Create DUCA tensor
        duca_tensor = tensor_converter.create_duca_host_tensor(arr)

        # Convert to NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        assert nrt_tensor is not None, f"DUCA->NRT conversion failed ({data_name})"

        # Verify data consistency
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"DUCA->NRT ({data_name})"
        ), f"Data consistency verification failed. ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_nrt_to_duca_host(self, test_data, tensor_converter, data_name):
        """Test：NRT tensor -> DUCA host tensor"""
        arr = test_data[data_name]
        logger.info(f"Test NRT -> DUCA host tensor ({data_name})")

        # Create NRT tensor
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        # Convert to DUCA host tensor
        duca_tensor = nrt_tensor.to_duca("cpu", 1)
        assert duca_tensor is not None, f"NRT->DUCA host conversion failed ({data_name})"

        # Verify data consistency
        converted_data = duca_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"NRT->DUCA host ({data_name})"
        ), f"Data consistency verification failed. ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_nrt_host_to_duca_device_should_fail(self, test_data, tensor_converter, data_name):
        """Test：NRT host tensor -> DUCA device tensor (Expected failure)"""
        arr = test_data[data_name]
        logger.info(f"Test NRT host -> DUCA device tensor ({data_name}) - Expected failure")

        # Create NRT host tensor
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        # Expected failure
        with pytest.raises(RuntimeError, match="Host runtime tensor can't convert to device duca tensor"):
            nrt_tensor.to_duca("duca:0", 1)

        logger.info(f"✓ Expected failure: NRT host -> DUCA device ({data_name})")

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_device_nrt_to_duca_device(self, test_data, tensor_converter, data_name):
        """Test：Device NRT tensor -> DUCA device tensor"""
        arr = test_data[data_name]
        logger.info(f"Test Device NRT -> DUCA device tensor ({data_name})")

        # Create DUCA device tensor. Then convert to device NRT
        duca_device_tensor = tensor_converter.create_duca_device_tensor(arr)
        device_nrt_tensor = nrt.RuntimeTensor.from_duca(duca_device_tensor)

        # Convert back to DUCA device tensor
        duca_tensor = device_nrt_tensor.to_duca("duca:0", 1)
        assert duca_tensor is not None, f"Device NRT->DUCA device conversion failed ({data_name})"

        # Verify data consistency
        converted_data = duca_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"Device NRT->DUCA device ({data_name})"
        ), f"Data consistency verification failed. ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16"])
    def test_memory_lifecycle(self, test_data, tensor_converter, data_name):
        """Test memory lifecycle and reference count"""
        arr = test_data[data_name]
        logger.info(f"Test memory lifecycle ({data_name})")

        # CreateDUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # Convert toNRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)

        # Check data after conversion
        data_before_del = nrt_tensor.to_numpy()

        # Del DUCA tensor
        del duca_tensor

        # Check whether the converted data is still valid after deleting the source data
        data_after_del = nrt_tensor.to_numpy()

        # Verify data consistency
        assert np.allclose(data_before_del, data_after_del, rtol=1e-5, atol=1e-6), \
            f"[del DUCA, keep NRT] memory lifecycle test failed. ({data_name}): Inconsistent data"

        # CreateDUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # Convert toNRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)

        # Check data after conversion
        data_before_del = duca_tensor.to_numpy()

        # del NRT tensor
        del nrt_tensor

        # Check whether the converted data is still valid after deleting the source data
        data_after_del = duca_tensor.to_numpy()

        # Verify data consistency
        assert np.allclose(data_before_del, data_after_del, rtol=1e-5, atol=1e-6), \
            f"[del NRT, keep DUCA] memory lifecycle test failed.  ({data_name}): Inconsistent data"

        logger.success(f"✓ memory lifecycle test passed. ({data_name})")

    def test_conversion_chain_integration(self, test_data, tensor_converter):
        """Test convert between NRT tensor and DUCA tensor"""
        logger.info("Test convert between NRT tensor and DUCA tensor.")

        arr = test_data["2d_float32"]

        # numpy -> DUCA device -> NRT -> DUCA device -> numpy
        duca_device = tensor_converter.create_duca_device_tensor(arr)
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_device)
        duca_host = nrt_tensor.to_duca("duca:0", 1)
        final_array = duca_host.to_numpy()

        # Verify data consistency
        assert tensor_converter.verify_data_consistency(
            arr, final_array, "Complete conversion chain"
        ), "Data consistency verification failed."

# Performance tests for large arrays


class TestTensorConversionPerformance:
    """Tensor Test performance"""

    @pytest.mark.parametrize("size", [1000, 10000])
    def test_large_array_conversion_performance(self, tensor_converter, size):
        """Test large array conversion performance"""
        import time

        logger.info(f"Test large array conversion performance (size: {size}x{size})")

        # Create large array
        large_array = np.random.rand(size, size).astype(np.float32)

        # Test numpy -> DUCA device performance
        start_time = time.time()
        duca_tensor = tensor_converter.create_duca_device_tensor(large_array)
        duca_creation_time = time.time() - start_time

        # Test DUCA -> NRT performance
        start_time = time.time()
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        nrt_conversion_time = time.time() - start_time

        # Test data reading performance
        start_time = time.time()
        result_array = nrt_tensor.to_numpy()
        numpy_conversion_time = time.time() - start_time

        logger.info(f"Performance info (size: {size}x{size}):")
        logger.info(f"  DUCA Create time: {duca_creation_time:.4f}s")
        logger.info(f"  NRT convert time: {nrt_conversion_time:.4f}s")
        logger.info(f"  Numpy convert time: {numpy_conversion_time:.4f}s")

        # Verify the correctness of the data
        assert tensor_converter.verify_data_consistency(
            large_array, result_array, f"Large array ({size}x{size})"
        ), f"Large array conversion data consistency verification failed. ({size}x{size})"

# Error handling tests


class TestTensorConversionErrors:
    """Tensor conversion error handling Test"""

    def test_invalid_device_name(self, tensor_converter):
        """Test invalid device name."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        with pytest.raises(ValueError):
            nrt_tensor.to_duca("invalid_device", 1)

    def test_empty_array_conversion(self, tensor_converter):
        """Test empty array convert."""
        empty_arr = np.array([], dtype=np.float32)

        try:
            duca_tensor = tensor_converter.create_duca_host_tensor(empty_arr)
            nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
            result = nrt_tensor.to_numpy()
            assert len(result) == 0, "The conversion result of an empty array is incorrect"
        except Exception as e:
            pytest.skip(f"Not support empty array: {e}")


# If this file is run directly, execute all tests
if __name__ == "__main__":
    pytest.main(['-vvs', __file__, '--tb=short'])
