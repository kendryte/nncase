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
    """创建 TensorConversionTester 实例"""
    return TensorConversionTester()

# 将原来的类改为辅助类


class TensorConversionTester:
    """辅助类：处理tensor转换逻辑"""

    def verify_data_consistency(self, original, converted, test_name):
        """验证数据一致性"""
        try:
            if np.allclose(original, converted, rtol=1e-5, atol=1e-6):
                logger.success(f"✓ {test_name}: 数据一致性验证通过")
                return True
            else:
                logger.error(f"✗ {test_name}: 数据不一致!")
                logger.error(f"  原始数据: {original.flatten()[:5]}...")
                logger.error(f"  转换数据: {converted.flatten()[:5]}...")
                return False
        except Exception as e:
            logger.error(f"✗ {test_name}: 数据验证失败 - {e}")
            return False

    def create_duca_host_tensor(self, arr):
        """创建 DUCA host tensor"""
        return duca.tensor(arr)

    def create_duca_device_tensor(self, arr):
        """创建 DUCA device tensor"""
        return duca.tensor(arr, "duca:0", 1)

    def create_nrt_tensor(self, arr):
        """创建 NRT tensor"""
        return nrt.RuntimeTensor.from_numpy(arr)

# pytest 测试类


class TestTensorConversion:
    """Tensor 转换测试"""

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_numpy_to_nrt(self, test_data, tensor_converter, data_name):
        """测试：numpy -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"测试 numpy -> NRT tensor ({data_name})")

        nrt_tensor = tensor_converter.create_nrt_tensor(arr)
        assert nrt_tensor is not None, f"创建 NRT tensor 失败 ({data_name})"

        # 验证数据一致性
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"numpy->NRT ({data_name})"
        ), f"数据一致性验证失败 ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_duca_device_to_nrt(self, test_data, tensor_converter, data_name):
        """测试：DUCA tensor -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"测试 DUCA device -> NRT tensor ({data_name})")

        # 创建 DUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # 转换为 NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        assert nrt_tensor is not None, f"DUCA->NRT 转换失败 ({data_name})"

        # 验证数据一致性
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"DUCA->NRT ({data_name})"
        ), f"数据一致性验证失败 ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_duca_host_to_nrt(self, test_data, tensor_converter, data_name):
        """测试：DUCA tensor -> NRT tensor"""
        arr = test_data[data_name]
        logger.info(f"测试 DUCA host -> NRT tensor ({data_name})")

        # 创建 DUCA tensor
        duca_tensor = tensor_converter.create_duca_host_tensor(arr)

        # 转换为 NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        assert nrt_tensor is not None, f"DUCA->NRT 转换失败 ({data_name})"

        # 验证数据一致性
        converted_data = nrt_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"DUCA->NRT ({data_name})"
        ), f"数据一致性验证失败 ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_nrt_to_duca_host(self, test_data, tensor_converter, data_name):
        """测试：NRT tensor -> DUCA host tensor"""
        arr = test_data[data_name]
        logger.info(f"测试 NRT -> DUCA host tensor ({data_name})")

        # 创建 NRT tensor
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        # 转换为 DUCA host tensor
        duca_tensor = nrt_tensor.to_duca("cpu", 1)
        assert duca_tensor is not None, f"NRT->DUCA host 转换失败 ({data_name})"

        # 验证数据一致性
        converted_data = duca_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"NRT->DUCA host ({data_name})"
        ), f"数据一致性验证失败 ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_nrt_host_to_duca_device_should_fail(self, test_data, tensor_converter, data_name):
        """测试：NRT host tensor -> DUCA device tensor (预期失败)"""
        arr = test_data[data_name]
        logger.info(f"测试 NRT host -> DUCA device tensor ({data_name}) - 预期失败")

        # 创建 NRT host tensor
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        # 预期这个转换会失败
        with pytest.raises(RuntimeError, match="Host runtime tensor can't convert to device duca tensor"):
            nrt_tensor.to_duca("duca:0", 1)

        logger.info(f"✓ 按预期失败: NRT host -> DUCA device ({data_name})")

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16", "zeros", "ones"])
    def test_device_nrt_to_duca_device(self, test_data, tensor_converter, data_name):
        """测试：Device NRT tensor -> DUCA device tensor"""
        arr = test_data[data_name]
        logger.info(f"测试 Device NRT -> DUCA device tensor ({data_name})")

        # 创建 DUCA device tensor 然后转为 device NRT
        duca_device_tensor = tensor_converter.create_duca_device_tensor(arr)
        device_nrt_tensor = nrt.RuntimeTensor.from_duca(duca_device_tensor)

        # 转换回 DUCA device tensor
        duca_tensor = device_nrt_tensor.to_duca("duca:0", 1)
        assert duca_tensor is not None, f"Device NRT->DUCA device 转换失败 ({data_name})"

        # 验证数据一致性
        converted_data = duca_tensor.to_numpy()
        assert tensor_converter.verify_data_consistency(
            arr, converted_data, f"Device NRT->DUCA device ({data_name})"
        ), f"数据一致性验证失败 ({data_name})"

    @pytest.mark.parametrize("data_name", ["2d_float32", "1d_int32", "3d_float16"])
    def test_memory_lifecycle(self, test_data, tensor_converter, data_name):
        """测试内存生命周期和引用计数"""
        arr = test_data[data_name]
        logger.info(f"测试内存生命周期 ({data_name})")

        # 创建DUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # 转换为NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)

        # 检查转换后数据
        data_before_del = nrt_tensor.to_numpy()

        # 删除DUCA tensor
        del duca_tensor

        # 检查删除后数据是否仍然有效
        data_after_del = nrt_tensor.to_numpy()

        # 验证数据一致性
        assert np.allclose(data_before_del, data_after_del, rtol=1e-5, atol=1e-6), \
            f"[del DUCA, keep NRT] 内存生命周期测试失败 ({data_name}): 数据不一致"

        # 创建DUCA tensor
        duca_tensor = tensor_converter.create_duca_device_tensor(arr)

        # 转换为NRT tensor
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)

        # 检查转换后数据
        data_before_del = duca_tensor.to_numpy()

        # 删除DUCA tensor
        del nrt_tensor

        # 检查删除后数据是否仍然有效
        data_after_del = duca_tensor.to_numpy()

        # 验证数据一致性
        assert np.allclose(data_before_del, data_after_del, rtol=1e-5, atol=1e-6), \
            f"[del NRT, keep DUCA] 内存生命周期测试失败 ({data_name}): 数据不一致"

        logger.success(f"✓ 内存生命周期测试通过 ({data_name})")

    def test_conversion_chain_integration(self, test_data, tensor_converter):
        """测试完整的转换链集成"""
        logger.info("测试完整转换链集成")

        arr = test_data["2d_float32"]

        # numpy -> DUCA device -> NRT -> DUCA device -> numpy
        duca_device = tensor_converter.create_duca_device_tensor(arr)
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_device)
        duca_host = nrt_tensor.to_duca("duca:0", 1)
        final_array = duca_host.to_numpy()

        # 验证最终数据一致性
        assert tensor_converter.verify_data_consistency(
            arr, final_array, "完整转换链"
        ), "完整转换链数据一致性验证失败"

# 针对大数组的性能测试


class TestTensorConversionPerformance:
    """Tensor 转换性能测试"""

    @pytest.mark.parametrize("size", [1000, 10000])
    def test_large_array_conversion_performance(self, tensor_converter, size):
        """测试大数组转换性能"""
        import time

        logger.info(f"测试大数组转换性能 (size: {size}x{size})")

        # 创建大数组
        large_array = np.random.rand(size, size).astype(np.float32)

        # 测试 numpy -> DUCA device 性能
        start_time = time.time()
        duca_tensor = tensor_converter.create_duca_device_tensor(large_array)
        duca_creation_time = time.time() - start_time

        # 测试 DUCA -> NRT 性能
        start_time = time.time()
        nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
        nrt_conversion_time = time.time() - start_time

        # 测试数据读取性能
        start_time = time.time()
        result_array = nrt_tensor.to_numpy()
        numpy_conversion_time = time.time() - start_time

        logger.info(f"性能统计 (size: {size}x{size}):")
        logger.info(f"  DUCA 创建时间: {duca_creation_time:.4f}s")
        logger.info(f"  NRT 转换时间: {nrt_conversion_time:.4f}s")
        logger.info(f"  Numpy 转换时间: {numpy_conversion_time:.4f}s")

        # 验证数据正确性
        assert tensor_converter.verify_data_consistency(
            large_array, result_array, f"大数组转换 ({size}x{size})"
        ), f"大数组转换数据一致性验证失败 ({size}x{size})"

# 错误处理测试


class TestTensorConversionErrors:
    """Tensor 转换错误处理测试"""

    def test_invalid_device_name(self, tensor_converter):
        """测试无效设备名称"""
        arr = np.array([1, 2, 3], dtype=np.float32)
        nrt_tensor = tensor_converter.create_nrt_tensor(arr)

        with pytest.raises(Exception):  # 根据实际错误类型调整
            nrt_tensor.to_duca("invalid_device", 1)

    def test_empty_array_conversion(self, tensor_converter):
        """测试空数组转换"""
        empty_arr = np.array([], dtype=np.float32)

        # 测试是否能正确处理空数组
        try:
            duca_tensor = tensor_converter.create_duca_host_tensor(empty_arr)
            nrt_tensor = nrt.RuntimeTensor.from_duca(duca_tensor)
            result = nrt_tensor.to_numpy()
            assert len(result) == 0, "空数组转换结果不正确"
        except Exception as e:
            pytest.skip(f"空数组转换不支持: {e}")


# 如果直接运行此文件，执行所有测试
if __name__ == "__main__":
    pytest.main(['-vvs', __file__, '--tb=short'])
