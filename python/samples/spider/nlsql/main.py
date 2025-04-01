from spider_env import SpiderEnv
from data_team import generate_query
import asyncio

async def main():
    done = False
    k = 0
    env = SpiderEnv()
    while not done and k < 1034:
        observation, info = env.reset(k=k)  # Dummy SQL query `state` contains the current benchmark item
        k = k + 1
        db_name = observation["observation"]
        question = observation["instruction"]
        schema = info["schema"]
        generated_query = await generate_query(question, schema, db_name)
        import re
        match = re.search(r'"sql":\s*"(.*)"', generated_query, re.DOTALL)
        if match:
            generated_query = match.group(1).strip()

        match = re.search(r"```sql\s*(.*?)\s*```", generated_query, re.DOTALL)
        if match:
            generated_query = match.group(1).strip()
        print("generated_query :" + str(generated_query))
        with open("/home/simges/autogen/spider/spider/nlsql/results/predicted_queries.txt", "a") as f:
            f.write(f"{generated_query}\n")

if __name__ == "__main__":
    asyncio.run(main())
