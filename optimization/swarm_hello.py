# swarm_hello.py
import modal

app = modal.App("swarm-hello")


@app.function()
def square(i: int) -> int:
    return i * i


@app.local_entrypoint()
def main():
    print(sum(square.map(range(200))))   # 200 inputs, run roughly in parallel in the cloud
