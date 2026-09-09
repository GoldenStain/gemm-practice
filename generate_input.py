if __name__ == "__main__":
    s = "\n".join(" ".join(["1"] * 1024) for _ in range(2048))
    with open("input.txt", "w") as f:
        f.write(s)
