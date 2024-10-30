def compute_adjacency_list(faces, num_vertices):
    adjacency_list = {i: set() for i in range(num_vertices)}
    for face in faces:
        i, j, k = face
        adjacency_list[i].update([j, k])
        adjacency_list[j].update([i, k])
        adjacency_list[k].update([i, j])
    # Convert sets to lists
    adjacency_list = {key: list(val) for key, val in adjacency_list.items()}
    return adjacency_list
