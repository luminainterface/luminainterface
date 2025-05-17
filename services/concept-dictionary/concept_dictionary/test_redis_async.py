import redis.asyncio as redis
import asyncio

async def test():
    r = redis.from_url('redis://:02211998@redis:6379', decode_responses=True)
    try:
        pong = await r.ping()
        print('Redis ping:', pong)
    except Exception as e:
        print('Redis connection error:', e)
    finally:
        await r.close()

if __name__ == "__main__":
    asyncio.run(test()) 