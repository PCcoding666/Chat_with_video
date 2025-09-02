"""
MiniCPM-V模型加载器 - Intel XPU优化版本
专门配置为使用Intel Arc GPU进行推理
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Dict, Any
import warnings


class MiniCPMVInference:
    """MiniCPM-V模型推理引擎 - Intel XPU版本 (INT4量化)"""
    
    def __init__(self, model_path: str = 'openbmb/MiniCPM-V-4_5-int4', device: str = 'xpu'):
        """
        初始化MiniCPM-V推理引擎
        
        Args:
            model_path: 模型路径，默认为MiniCPM-V-4.5-int4量化版本
            device: 设备类型，默认为'xpu'（Intel GPU）
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # Intel GPU环境变量设置
        self._setup_intel_gpu_env()
        
        print(f"MiniCPM-V推理引擎已配置 (INT4量化版本)")
        print(f"模型路径: {model_path}")
        print(f"目标设备: {device}")
    
    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return
            
        # 验证XPU可用性
        if not self._check_xpu_availability():
            raise RuntimeError("Intel XPU不可用，请检查驱动和环境配置")
        
        print(f"正在初始化MiniCPM-V推理引擎...")
        print(f"XPU设备数量: {torch.xpu.device_count()}")
        
        # 加载模型和分词器
        self._load_model()
        self._load_tokenizer()
        
        self._initialized = True
        print("模型初始化完成!")
    
    def _setup_intel_gpu_env(self):
        """设置Intel GPU相关环境变量"""
        env_vars = {
            'SYCL_CACHE_PERSISTENT': '1',
            'ZE_AFFINITY_MASK': '0',
            'USE_XPU': '1'
        }
        
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                print(f"设置环境变量: {key}={value}")
    
    def _check_xpu_availability(self) -> bool:
        """检查Intel XPU设备可用性 - 带DLL依赖绕过"""
        try:
            # 首先尝试绕过可能的DLL依赖问题
            print("尝试绕过Intel XPU DLL依赖问题...")
            
            # 方法1: 延迟导入torch.xpu
            try:
                import torch.xpu
                is_available = torch.xpu.is_available()
            except OSError as dll_error:
                if "126" in str(dll_error) or "c10_xpu.dll" in str(dll_error):
                    print(f"⚠️ Intel XPU DLL依赖问题: {dll_error}")
                    print("尝试使用CPU模式作为回退...")
                    self.device = 'cpu'
                    return False
                else:
                    raise dll_error
            
            if is_available:
                device_count = torch.xpu.device_count()
                current_device = torch.xpu.current_device()
                print(f"XPU状态: 可用")
                print(f"设备数量: {device_count}")
                print(f"当前设备: {current_device}")
                
                # 测试简单的XPU操作
                test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('xpu')
                result = test_tensor.sum()
                print(f"XPU测试: {result.item()}")
                
                return True
            else:
                print("XPU状态: 不可用")
                print("回退到CPU模式...")
                self.device = 'cpu'
                return False
        except Exception as e:
            print(f"XPU检查失败: {str(e)}")
            print("强制回退到CPU模式...")
            self.device = 'cpu'
            return False
    
    def _load_model(self):
        """加载MiniCPM-V模型 - INT4量化版本强制XPU加载策略 (终极版本)"""
        try:
            print("正在加载INT4量化模型到XPU...")
            
            # 清理XPU缓存
            self._clear_xpu_cache()
            
            # 强制XPU加载 - 绕过Transformers的显存查询机制
            if self.device != 'xpu':
                print(f"⚠️ 设备不是XPU ({self.device})，强制设置为XPU")
                self.device = 'xpu'
            
            print("🚀 使用终极策略彻底绕过Intel XPU显存查询问题...")
            
            # 策略1: 全面禁用PyTorch的显存查询功能
            print("Step 1: 彻底禁用PyTorch显存查询机制...")
            
            # 保存原始函数的引用
            self._patch_pytorch_memory_functions()
            
            try:
                # 策略2: 强制设置环境变量
                print("Step 2: 设置强制XPU环境变量...")
                original_env = self._setup_force_xpu_env()
                
                # 策略3: 分阶段加载模型
                print("Step 3: 分阶段加载模型...")
                
                # 3.1 先加载配置
                print("  3.1 加载模型配置...")
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # 3.2 强制配置torch_dtype
                config.torch_dtype = torch.float16
                
                # 3.3 加载模型到CPU
                print("  3.2 在CPU上加载模型...")
                with torch.device('cpu'):
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map=None,  # 完全禁用自动设备分配
                        max_memory=None,  # 禁用内存管理
                    )
                
                # 3.4 逐层移动到XPU
                print("  3.3 逐层移动模型到XPU...")
                self._move_model_to_xpu_safely()
                
                print("✅ 成功！INT4模型已彻底加载到Intel XPU")
                
                # 恢复环境变量
                self._restore_env(original_env)
                
            finally:
                # 恢复所有被修改的函数
                self._restore_pytorch_memory_functions()
            
            # 设置为评估模式
            self.model = self.model.eval()
            
            # 验证模型设备
            self._verify_model_device()
            
            # 显示模型设备分布
            self._print_device_distribution()
            
            # 显示XPU使用情况（如果可能）
            try:
                if hasattr(torch.xpu, 'memory_allocated'):
                    allocated = torch.xpu.memory_allocated() / 1024**3
                    print(f"XPU显存使用: {allocated:.2f} GB")
            except Exception as mem_e:
                print(f"无法查询XPU显存使用: {mem_e}")
            
            print(f"✅ INT4量化模型强制加载完成 - 运行在: {self.device.upper()}")
            print("🎉 已彻底绕过Intel XPU显存查询限制！")
            
        except Exception as e:
            print(f"XPU强制加载失败: {str(e)}")
            import traceback
            print("\n详细错误信息:")
            traceback.print_exc()
            raise
    
    def _load_tokenizer(self):
        """加载分词器"""
        try:
            print("正在加载分词器...")
            
            # 尝试多种加载方式
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"直接加载失败，尝试使用模型对象的tokenizer: {e}")
                # 如果直接加载失败，尝试从模型对象获取
                if hasattr(self.model, 'tokenizer'):
                    self.tokenizer = self.model.tokenizer
                    print("从模型对象获取tokenizer成功")
                else:
                    # 如果还是失败，尝试使用Qwen2的tokenizer作为备选
                    print("尝试使用备选tokenizer...")
                    from transformers import Qwen2Tokenizer
                    self.tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
                    print("使用备选tokenizer成功")
            
            print("分词器加载完成")
            
        except Exception as e:
            print(f"分词器加载失败: {str(e)}")
            print("将使用空的tokenizer，可能影响模型功能")
            self.tokenizer = None
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': self.device,
            'xpu_available': torch.xpu.is_available(),
            'device_count': torch.xpu.device_count() if torch.xpu.is_available() else 0,
        }
        
        if torch.xpu.is_available():
            info['current_device'] = torch.xpu.current_device()
            
            # 内存信息（如果可用）
            try:
                if hasattr(torch.xpu, 'memory_allocated'):
                    info['memory_allocated_gb'] = torch.xpu.memory_allocated() / 1024**3
                if hasattr(torch.xpu, 'memory_reserved'):
                    info['memory_reserved_gb'] = torch.xpu.memory_reserved() / 1024**3
            except:
                pass
        
        return info
    
    def chat(self, msgs: List[Dict], use_image_id: bool = False, 
             max_slice_nums: int = 1, temporal_ids: Optional[List[List[int]]] = None,
             max_new_tokens: int = 2048, do_sample: bool = True, 
             temperature: float = 0.7, top_p: float = 0.8) -> str:
        """
        与模型进行对话
        
        Args:
            msgs: 消息列表，包含图像和文本
            use_image_id: 是否使用图像ID
            max_slice_nums: 最大切片数量
            temporal_ids: 时序ID列表（用于视频）
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            temperature: 温度参数
            top_p: Top-p采样参数
            
        Returns:
            模型的回答文本
        """
        # 确保模型已初始化
        if not self._initialized:
            self.initialize()
            
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("模型或分词器未正确加载")
            
            print("开始推理...")
            
            # 禁用梯度计算以节省内存
            with torch.no_grad():
                # 生成参数
                generation_config = {
                    'max_new_tokens': max_new_tokens,
                    'do_sample': do_sample,
                    'temperature': temperature,
                    'top_p': top_p,
                    'use_cache': True,
                }
                
                # 如果有时序ID，添加到参数中
                if temporal_ids is not None:
                    generation_config['temporal_ids'] = temporal_ids
                
                # 调用模型的chat方法
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    use_image_id=use_image_id,
                    max_slice_nums=max_slice_nums,
                    **generation_config
                )
                
                print("推理完成")
                return answer
                
        except Exception as e:
            print(f"推理失败: {str(e)}")
            raise
    
    def warm_up(self):
        """模型预热，预加载CUDA kernel"""
        try:
            print("正在预热模型...")
            
            # 创建虚拟输入进行预热
            dummy_msgs = [
                {'role': 'user', 'content': ['Hello, this is a warm-up message.']}
            ]
            
            with torch.no_grad():
                _ = self.model.chat(
                    msgs=dummy_msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            print("模型预热完成")
            
        except Exception as e:
            print(f"模型预热失败: {str(e)}")
    
    def _clear_xpu_cache(self):
        """清理XPU缓存"""
        try:
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
                print("XPU缓存已清理")
            # 强制垃圾回收
            import gc
            gc.collect()
        except Exception as e:
            print(f"缓存清理失败: {str(e)}")
    
    def _verify_model_device(self):
        """验证模型是否在正确的设备上"""
        try:
            # 检查模型的第一个参数所在设备
            first_param = next(self.model.parameters())
            actual_device = str(first_param.device)
            
            if self.device == 'xpu' and 'xpu' not in actual_device:
                print(f"⚠️ 警告: 期望设备 {self.device}，但模型在 {actual_device}")
                return False
            else:
                print(f"✅ 模型成功加载到设备: {actual_device}")
                return True
                
        except Exception as e:
            print(f"设备验证失败: {str(e)}")
            return False
    
    def _print_device_distribution(self):
        """打印模型设备分布"""
        try:
            device_map = {}
            total_params = 0
            
            for name, param in self.model.named_parameters():
                device = str(param.device)
                param_count = param.numel()
                total_params += param_count
                
                if device not in device_map:
                    device_map[device] = {'count': 0, 'params': 0}
                device_map[device]['count'] += 1
                device_map[device]['params'] += param_count
            
            print("模型设备分布详情:")
            for device, info in device_map.items():
                percentage = (info['params'] / total_params) * 100 if total_params > 0 else 0
                print(f"  {device}: {info['count']} 层, {info['params']:,} 参数 ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"设备分布显示失败: {str(e)}")
    
    def _patch_pytorch_memory_functions(self):
        """修补PyTorch的内存查询函数以避免XPU显存查询错误"""
        try:
            print("    正在修补PyTorch内存查询函数...")
            
            # 保存原始函数引用
            self._original_functions = {}
            
            # 1. 修补 torch.cuda.mem_get_info (如果存在)
            if hasattr(torch.cuda, 'mem_get_info'):
                self._original_functions['cuda_mem_get_info'] = torch.cuda.mem_get_info
                torch.cuda.mem_get_info = lambda device=None: (0, 0)
            
            # 2. 修补 torch.xpu.mem_get_info (如果存在)
            try:
                if hasattr(torch.xpu, 'mem_get_info'):
                    self._original_functions['xpu_mem_get_info'] = torch.xpu.mem_get_info
                    torch.xpu.mem_get_info = lambda device=None: (0, 0)
            except:
                pass
            
            # 3. 修补 transformers 中的相关函数
            import transformers.utils
            import transformers.modeling_utils
            
            # 修补 get_available_memory 相关函数
            if hasattr(transformers.utils, 'get_available_memory'):
                self._original_functions['get_available_memory'] = transformers.utils.get_available_memory
                transformers.utils.get_available_memory = lambda: {}
            
            # 修补 caching_allocator_warmup
            if hasattr(transformers.modeling_utils, 'caching_allocator_warmup'):
                self._original_functions['caching_allocator_warmup'] = transformers.modeling_utils.caching_allocator_warmup
                def dummy_warmup(*args, **kwargs):
                    pass
                transformers.modeling_utils.caching_allocator_warmup = dummy_warmup
            
            # 4. 修补 accelerate 库的内存函数
            try:
                import accelerate.utils
                if hasattr(accelerate.utils, 'get_available_memory'):
                    self._original_functions['accelerate_get_available_memory'] = accelerate.utils.get_available_memory
                    accelerate.utils.get_available_memory = lambda: {}
            except ImportError:
                pass  # accelerate 可能未安装
            
            print(f"    ✅ 已修补 {len(self._original_functions)} 个内存查询函数")
            
        except Exception as e:
            print(f"    ❌ 修补函数失败: {e}")
            self._original_functions = {}
    
    def _restore_pytorch_memory_functions(self):
        """恢复被修补的PyTorch内存查询函数"""
        try:
            if not hasattr(self, '_original_functions'):
                return
            
            print("    正在恢复PyTorch内存查询函数...")
            
            # 恢复 torch.cuda.mem_get_info
            if 'cuda_mem_get_info' in self._original_functions:
                torch.cuda.mem_get_info = self._original_functions['cuda_mem_get_info']
            
            # 恢复 torch.xpu.mem_get_info
            if 'xpu_mem_get_info' in self._original_functions:
                torch.xpu.mem_get_info = self._original_functions['xpu_mem_get_info']
            
            # 恢复 transformers 函数
            import transformers.utils
            import transformers.modeling_utils
            
            if 'get_available_memory' in self._original_functions:
                transformers.utils.get_available_memory = self._original_functions['get_available_memory']
            
            if 'caching_allocator_warmup' in self._original_functions:
                transformers.modeling_utils.caching_allocator_warmup = self._original_functions['caching_allocator_warmup']
            
            # 恢复 accelerate 函数
            if 'accelerate_get_available_memory' in self._original_functions:
                try:
                    import accelerate.utils
                    accelerate.utils.get_available_memory = self._original_functions['accelerate_get_available_memory']
                except ImportError:
                    pass
            
            print(f"    ✅ 已恢复 {len(self._original_functions)} 个函数")
            self._original_functions = {}
            
        except Exception as e:
            print(f"    ❌ 恢复函数失败: {e}")
    
    def _setup_force_xpu_env(self):
        """设置强制XPU环境变量"""
        original_env = {}
        
        force_env = {
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
            'CUDA_LAUNCH_BLOCKING': '0',
            'PYTORCH_NO_CUDA_MEMORY_CACHING': '0',
            'INTEL_XPU_FORCE_LOAD': '1',
            'XPU_FORCE_DEVICE_ALLOC': '1',
            'SYCL_DISABLE_MEM_QUERY': '1',
        }
        
        for key, value in force_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
            print(f"    设置环境变量: {key}={value}")
        
        return original_env
    
    def _restore_env(self, original_env):
        """恢复环境变量"""
        for key, value in original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    def _move_model_to_xpu_safely(self):
        """安全地将模型逐层移动到XPU"""
        try:
            print("    开始逐层移动模型到XPU...")
            
            # 统计层数
            layer_count = 0
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # 叶子模块
                    layer_count += 1
            
            print(f"    总共需要移动 {layer_count} 个模块")
            
            # 逐层移动
            moved_count = 0
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # 叶子模块
                    try:
                        # 检查模块是否有参数
                        has_params = any(p.numel() > 0 for p in module.parameters())
                        if has_params:
                            # 使用 torch.device 上下文确保正确移动
                            with torch.device('xpu'):
                                module.to('xpu')
                            moved_count += 1
                            
                            if moved_count % 50 == 0:  # 每50层显示一次进度
                                print(f"    已移动 {moved_count}/{layer_count} 个模块")
                    except Exception as e:
                        print(f"    ⚠️ 模块 {name} 移动失败: {e}，继续处理其他模块...")
                        continue
            
            # 最终确保整个模型在XPU上
            self.model = self.model.to('xpu')
            print(f"    ✅ 成功移动 {moved_count} 个模块到XPU")
            
        except Exception as e:
            print(f"    ❌ 模型移动失败: {e}")
            # 如果逐层移动失败，尝试整体移动
            print("    尝试整体移动模型...")
            self.model = self.model.to('xpu')
    
    def clear_cache(self):
        """清理XPU缓存"""
        self._clear_xpu_cache()
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self.clear_cache()
        except:
            pass


if __name__ == "__main__":
    # 测试模型加载
    try:
        print("测试MiniCPM-V模型加载器...")
        
        inference_engine = MiniCPMVInference()
        
        # 显示设备信息
        device_info = inference_engine.get_device_info()
        print("设备信息:", device_info)
        
        # 预热模型
        inference_engine.warm_up()
        
        # 测试简单对话
        test_msgs = [
            {'role': 'user', 'content': ['你好，这是一个测试消息。']}
        ]
        
        response = inference_engine.chat(test_msgs, max_new_tokens=50)
        print("测试回答:", response)
        
        print("模型加载器测试成功!")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")