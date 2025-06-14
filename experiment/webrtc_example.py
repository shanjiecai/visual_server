import time

from experiment.webrtc_client import WebRTCStreamClient
import cv2


def main():
    # 创建客户端实例
    client = WebRTCStreamClient(url="http://112.13.96.171:1081/api/call/whep/get?src=1830004229212345670000034")

    # 启动客户端
    client.start()
    print("客户端已启动")

    # 等待连接建立
    print("等待连接建立...")
    time.sleep(1)

    try:
        # 算法处理循环
        while True:
            # 获取最新帧
            frame = client.get_latest_frame()

            if frame is not None:
                # 在这里添加算法的处理逻辑


                cv2.imshow('Latest Frame', frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 控制处理频率
            time.sleep(0.03)  # 约30FPS

    except KeyboardInterrupt:
        print("正在停止...")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 停止客户端
        client.stop()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()