#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频流处理系统主入口

提供统一的启动接口，支持不同的运行模式：

运行模式：
- demo: 演示模式，运行摄像头处理流水线
- worker: 工作进程模式，运行VLM工作进程
- both: 同时运行演示程序和工作进程

用法:
    python main.py demo                         # 运行演示程序
    python main.py worker                       # 运行VLM工作进程
    python main.py both                         # 同时运行两者
    python main.py demo --config config.yaml   # 使用指定配置
"""

import asyncio
import argparse
import sys
import subprocess
import threading
import os
from pathlib import Path
from loguru import logger


def setup_logging():
    """配置日志系统"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )


# ==================== 运行模式实现 ====================

async def run_demo(config_path: str = None):
    """运行演示程序
    
    Args:
        config_path: 配置文件路径，None使用默认配置
    """
    from demo import main as demo_main
    
    # 临时修改sys.argv以传递配置参数
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["demo.py"]
        if config_path:
            sys.argv.extend(["--config", config_path])
        
        logger.info("启动演示程序...")
        await demo_main()
    finally:
        sys.argv = original_argv


def run_worker(config_path: str = None):
    """运行VLM工作进程
    
    Args:
        config_path: 配置文件路径，None使用默认配置
    """
    # 确定配置文件路径
    if not config_path:
        config_path = "worker/vlm_worker_config.yaml"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        logger.error(f"VLM工作进程配置文件不存在: {config_path}")
        logger.info("请确保配置文件存在或使用 --config 参数指定配置文件")
        return
    
    # 构建启动命令
    cmd = [sys.executable, "-m", "worker.vlm_worker", "--config", config_path]
    logger.info(f"启动VLM工作进程: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("VLM工作进程被用户中断")
    except subprocess.CalledProcessError as e:
        logger.error(f"VLM工作进程异常退出: {e}")
    except Exception as e:
        logger.error(f"启动VLM工作进程失败: {e}")


async def run_both(config_path: str = None):
    """同时运行演示程序和工作进程
    
    Args:
        config_path: 配置文件路径，None使用默认配置
    """
    logger.info("同时启动演示程序和VLM工作进程...")
    
    # 在单独线程中运行worker（因为它是同步的）
    worker_thread = threading.Thread(
        target=run_worker,
        args=(config_path,),
        daemon=True,
        name="vlm_worker_thread"
    )
    
    try:
        # 启动worker线程
        worker_thread.start()
        logger.info("VLM工作进程线程已启动")
        
        # 在主线程中运行demo（异步）
        await run_demo(config_path)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止所有组件...")
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
    finally:
        # 等待worker线程结束（如果还在运行）
        if worker_thread.is_alive():
            logger.info("等待VLM工作进程线程结束...")
            worker_thread.join(timeout=5)  # 最多等待5秒


# ==================== 命令行处理 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="视频流处理系统主入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  demo     运行演示程序（摄像头处理流水线）
  worker   运行VLM工作进程
  both     同时运行演示程序和工作进程

示例:
  python main.py demo                         # 运行演示程序
  python main.py worker                       # 运行VLM工作进程
  python main.py both                         # 同时运行两者
  python main.py demo --config config.yaml   # 使用指定配置
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["demo", "worker", "both"],
        help="运行模式"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )
    
    return parser.parse_args()


def validate_config_file(config_path: str, mode: str) -> bool:
    """验证配置文件
    
    Args:
        config_path: 配置文件路径
        mode: 运行模式
        
    Returns:
        配置文件是否有效
    """
    if not config_path:
        return True  # 使用默认配置
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return False
    
    # 检查文件扩展名
    if not config_path.endswith(('.yaml', '.yml')):
        logger.warning(f"配置文件可能不是YAML格式: {config_path}")
    
    logger.info(f"使用配置文件: {config_path}")
    return True


async def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger.info("视频流处理系统启动")
    
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 验证配置文件
        if not validate_config_file(args.config, args.mode):
            sys.exit(1)
        
        # 根据模式运行相应功能
        if args.mode == "demo":
            await run_demo(args.config)
        elif args.mode == "worker":
            # worker是同步函数，需要在线程中运行
            worker_thread = threading.Thread(
                target=run_worker,
                args=(args.config,),
                name="vlm_worker_main"
            )
            worker_thread.start()
            worker_thread.join()  # 等待worker结束
        elif args.mode == "both":
            await run_both(args.config)
        else:
            logger.error(f"不支持的运行模式: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("视频流处理系统已退出")


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main()) 