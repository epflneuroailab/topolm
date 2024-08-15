for s in moseley elli
do
    echo "$s | r = 10 | a = 2.5 | decay = 0"
    python3 responses.py radius=10 neighborhoods=5 alpha=2.5 accum=mean decay=0 batch_size=24 stimulus=$s
    python3 visualize.py radius=10 neighborhoods=5 alpha=2.5 accum=mean decay=0 batch_size=24 stimulus=$s

    echo "$s | r = 5 | a = 2.5 | decay = 0"
    python3 responses.py radius=5 neighborhoods=5 alpha=2.5 accum=mean decay=0 batch_size=24 stimulus=$s
    python3 visualize.py radius=5 neighborhoods=5 alpha=2.5 accum=mean decay=0 batch_size=24 stimulus=$s

    echo "$s | r = 10 | a = 1.0 | decay = 0.0001"
    python3 responses.py radius=10 neighborhoods=5 alpha=1.0 accum=mean decay=0.0001 batch_size=24 stimulus=$s
    python3 visualize.py radius=10 neighborhoods=5 alpha=1.0 accum=mean decay=0.0001 batch_size=24 stimulus=$s
done