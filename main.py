#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频流处理系统主入口和管理工具

提供统一的启动接口，支持不同的运行模式和进程管理：

运行模式：
- demo: 演示模式，运行摄像头处理流水线
- worker: 工作进程模式，运行VLM工作进程
- both: 同时运行演示程序和工作进程

进程管理：
- start: 启动组件（后台运行）
- stop: 停止组件
- status: 查看状态
- run: 前台运行（开发调试用）

用法:
    # 前台运行（开发调试）
    python main.py run demo                    # 前台运行演示程序
    python main.py run worker                  # 前台运行VLM工作进程
    python main.py run both                    # 前台同时运行两者
    
    # 后台运行（生产环境）
    python main.py start demo                  # 后台启动演示程序
    python main.py start worker               # 后台启动VLM工作进程
    python main.py stop                       # 停止所有组件
    python main.py stop demo                  # 停止指定组件
    python main.py status                     # 查看运行状态
    
    # 使用指定配置
    python main.py run demo --config config.yaml
    python main.py start worker --config worker/custom_config.yaml
"""

import asyncio
import argparse
import sys
import subprocess
import threading
import os
import time
import signal
import json
import psutil
from pathlib import Path
from loguru import logger


# 进程信息保存路径
PROCESS_INFO_FILE = "process_info.json"
LOG_DIR = Path("logs")


def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )


# ==================== 前台运行功能 ====================

async def run_demo(config_path=None):
    """前台运行演示程序"""
    from demo import main as demo_main
    
    # 临时修改sys.argv以传递配置参数
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["demo.py"]
        if config_path:
            sys.argv.extend(["--config", config_path])
        
        await demo_main()
    finally:
        sys.argv = original_argv


def run_worker(config_path=None):
    """前台运行VLM工作进程"""
    cmd = [sys.executable, "-m", "worker.vlm_worker"]
    if config_path:
        cmd.extend(["--config", config_path])
    else:
        cmd.extend(["--config", "worker/vlm_worker_config.yaml"])
    
    logger.info(f"启动VLM工作进程: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("VLM工作进程被用户中断")
    except subprocess.CalledProcessError as e:
        logger.error(f"VLM工作进程异常退出: {e}")


async def run_both(config_path=None):
    """前台同时运行演示程序和工作进程"""
    # 在单独的线程中运行worker
    worker_thread = threading.Thread(
        target=run_worker,
        args=(config_path,),
        daemon=True
    )
    worker_thread.start()
    
    # 在主线程中运行demo
    await run_demo(config_path)


# ==================== 后台进程管理功能 ====================

def save_process_info(component, pid, cmd):
    """保存进程信息到文件"""
    LOG_DIR.mkdir(exist_ok=True)
    
    if os.path.exists(PROCESS_INFO_FILE):
        with open(PROCESS_INFO_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    
    data[component] = {"pid": pid, "cmd": cmd, "start_time": time.time()}
    
    with open(PROCESS_INFO_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_process_info():
    """从文件加载进程信息"""
    if not os.path.exists(PROCESS_INFO_FILE):
        return {}
    
    try:
        with open(PROCESS_INFO_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def is_process_running(pid):
    """检查进程是否正在运行"""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def start_demo_background(config_path=None):
    """后台启动演示程序"""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "demo.log"
    
    cmd = [sys.executable, "demo.py"]
    if config_path:
        cmd.extend(["--config", config_path])
    
    print(f"后台启动演示程序: {' '.join(cmd)}")
    
    with open(log_file, 'a') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True
        )
    
    save_process_info("demo", process.pid, ' '.join(cmd))
    print(f"演示程序已启动 (PID: {process.pid})")
    print(f"日志文件: {log_file}")


def start_worker_background(config_path=None):
    """后台启动VLM工作进程"""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "vlm_worker.log"
    
    cmd = [sys.executable, "-m", "worker.vlm_worker"]
    if config_path:
        cmd.extend(["--config", config_path])
    else:
        cmd.extend(["--config", "worker/vlm_worker_config.yaml"])
    
    print(f"后台启动VLM工作进程: {' '.join(cmd)}")
    
    with open(log_file, 'a') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True
        )
    
    save_process_info("worker", process.pid, ' '.join(cmd))
    print(f"VLM工作进程已启动 (PID: {process.pid})")
    print(f"日志文件: {log_file}")


def stop_process(component):
    """停止进程"""
    process_info = load_process_info()
    
    if component not in process_info:
        print(f"{component} 未在运行")
        return
    
    pid = process_info[component]["pid"]
    
    if not is_process_running(pid):
        print(f"{component} 已停止 (PID: {pid})")
        # 清理进程信息
        del process_info[component]
        with open(PROCESS_INFO_FILE, 'w') as f:
            json.dump(process_info, f, indent=2)
        return
    
    print(f"停止 {component} (PID: {pid})...")
    
    try:
        # 尝试优雅停止
        os.kill(pid, signal.SIGTERM)
        
        # 等待最多10秒
        for _ in range(10):
            if not is_process_running(pid):
                print(f"{component} 已停止")
                break
            time.sleep(1)
        else:
            # 强制终止
            print(f"{component} 未响应，强制终止...")
            os.kill(pid, signal.SIGKILL)
            print(f"{component} 已强制终止")
    
    except ProcessLookupError:
        print(f"{component} 已停止")
    
    # 更新进程信息
    process_info = load_process_info()
    if component in process_info:
        del process_info[component]
        with open(PROCESS_INFO_FILE, 'w') as f:
            json.dump(process_info, f, indent=2)


def stop_all():
    """停止所有进程"""
    process_info = load_process_info()
    if not process_info:
        print("没有运行的组件")
        return
    
    for component in list(process_info.keys()):
        stop_process(component)


def show_status():
    """显示所有组件状态"""
    process_info = load_process_info()
    
    if not process_info:
        print("没有运行的组件")
        return
    
    print("组件状态:")
    print("-" * 70)
    print(f"{'组件':<15} {'PID':<10} {'状态':<10} {'启动时间':<20} {'日志文件':<15}")
    print("-" * 70)
    
    for component, info in process_info.items():
        pid = info["pid"]
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["start_time"]))
        log_file = f"logs/{component}.log"
        
        if is_process_running(pid):
            status = "运行中"
        else:
            status = "已停止"
        
        print(f"{component:<15} {pid:<10} {status:<10} {start_time:<20} {log_file:<15}")


# ==================== 命令行解析和主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频流处理系统管理工具")
    
    subparsers = parser.add_subparsers(dest="action", help="操作类型")
    
    # run 子命令（前台运行）
    run_parser = subparsers.add_parser("run", help="前台运行组件（开发调试用）")
    run_parser.add_argument("component", choices=["demo", "worker", "both"], help="组件名称")
    run_parser.add_argument("--config", type=str, help="配置文件路径")
    
    # start 子命令（后台启动）
    start_parser = subparsers.add_parser("start", help="后台启动组件")
    start_parser.add_argument("component", choices=["demo", "worker"], help="组件名称")
    start_parser.add_argument("--config", type=str, help="配置文件路径")
    
    # stop 子命令（停止）
    stop_parser = subparsers.add_parser("stop", help="停止组件")
    stop_parser.add_argument("component", nargs="?", choices=["demo", "worker"], help="组件名称（不指定则停止所有）")
    
    # status 子命令（状态查看）
    subparsers.add_parser("status", help="查看组件运行状态")
    
    return parser.parse_args()


async def main():
    """主函数"""
    setup_logging()
    args = parse_args()
    
    if not args.action:
        print("请指定操作类型，使用 --help 查看帮助")
        return
    
    try:
        if args.action == "run":
            # 前台运行
            logger.info(f"前台启动视频流处理系统 - 组件: {args.component}")
            
            if args.component == "demo":
                await run_demo(args.config)
            elif args.component == "worker":
                run_worker(args.config)
            elif args.component == "both":
                await run_both(args.config)
        
        elif args.action == "start":
            # 后台启动
            print(f"后台启动组件: {args.component}")
            
            if args.component == "demo":
                start_demo_background(args.config)
            elif args.component == "worker":
                start_worker_background(args.config)
        
        elif args.action == "stop":
            # 停止组件
            if args.component:
                stop_process(args.component)
            else:
                stop_all()
        
        elif args.action == "status":
            # 查看状态
            show_status()
    
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 