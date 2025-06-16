from aiokafka import AIOKafkaProducer
import asyncio


async def produce():
    # 初始化生产者，注意 bootstrap_servers 要指向你的 Kafka 地址
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',  # 如果从宿主机访问，用 localhost:9092
        # 如果从其他容器访问，用 kafka:29092 (对应 docker-compose 中的服务名)
    )
    
    # 启动生产者
    await producer.start()
    
    try:
        # 发送消息
        for i in range(5):
            message = f"message-{i}".encode('utf-8')
            await producer.send_and_wait(
                topic="test_topic",  # 确保这个主题已存在
                value=message
            )
            print(f"Sent: {message}")
    finally:
        # 关闭生产者
        await producer.stop()


# 运行生产消息
asyncio.run(produce())

