def split(samples, n):
    for i in range(0, len(samples), n):
        yield samples[i:i + n]