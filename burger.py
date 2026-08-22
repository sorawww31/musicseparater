import asyncio
import time


def slow_job(name):
    print(f"{name}: Starting...")
    time.sleep(3)
    print(f"{name}: Completed!")


def AtoB():
    slow_job("A")
    slow_job("B")

async def async_slow_job(name):
    print(f"{name}: Starting...")
    # time.sleep(3)
    await asyncio.sleep(3)
    return (f"{name}: Completed!")


async def await_AtoB():
    result1 = await async_slow_job("A")
    print(result1)
    result2 = await async_slow_job("B")
    print(result2)
    print("Completed All Jobs!")

async def create_tasks():
    # create_task()を使うと、タスクを作成してすぐに実行することができる
    # await task1にすると、task1を完了するまで待機する
    # その間、create_task()で作成したtask2も実行される
    # task2は、先に実行させておいて、あとからawaitで結果を取得している。
    task1 = asyncio.create_task(async_slow_job("A"))
    task2 = asyncio.create_task(async_slow_job("B"))
    result1 = await task1
    print(result1)
    result2 = await task2
    print(result2)
    print("Completed All Jobs!")

if __name__ == "__main__":
    # AtoB()
    # async_slow_job("A")
    # 一番外側のasync関数を呼び出すときはasyncio.run()を使う
    # asyncio.run(await_AtoB()) await→awaitなのでまだ直立
    # asyncio.run(await_AtoB())
    asyncio.run(create_tasks())