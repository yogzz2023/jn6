def group_measurements_into_tracks(measurements):
    """Group measurements into tracks based on range difference condition."""
    tracks = []
    used_indices = set()
    
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        
        # Initialize track ID
        track_id = len(tracks) + 1
        
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            
            # Calculate range difference
            range_diff = abs(np.linalg.norm(np.array([x, y, z]) - np.array([x_base, y_base, z_base])))
            
            # Check if range difference is within a threshold (e.g., 1 km)
            if range_diff <= 1.0:
                track.append((x, y, z, mt))
                used_indices.add(j)
        
        # Assign track ID to measurements in the track
        for k in range(len(track)):
            track[k] = (*track[k], track_id)
        
        tracks.append(track)
    
    return tracks

def perform_clustering(tracks):
    """Perform clustering within each track."""
    all_clusters = []
    for track in tracks:
        clusters = []
        for measurement in track:
            cluster = [measurement]
            for other_measurement in track:
                if other_measurement != measurement:
                    range_diff = abs(np.linalg.norm(np.array(measurement[:3]) - np.array(other_measurement[:3])))
                    if range_diff <= 1.0:  # Adjust threshold as needed
                        cluster.append(other_measurement)
            clusters.append(cluster)
        all_clusters.append(clusters)
    return all_clusters

def generate_hypotheses_and_probabilities(clusters, cov_inv):
    """Generate hypotheses and calculate probabilities for each track."""
    all_hypotheses = []
    all_probabilities = []
    for track_clusters in clusters:
        hypotheses = []
        probabilities = []
        for cluster in track_clusters:
            num_measurements = len(cluster)
            if num_measurements > 1:  # At least two measurements needed for hypothesis generation
                base = num_measurements + 1
                for count in range(base ** num_measurements):
                    hypothesis = []
                    for idx in range(num_measurements):
                        report_idx = (count // (base ** idx)) % base
                        hypothesis.append((cluster[idx], report_idx - 1))

                    if is_valid_hypothesis(hypothesis):
                        hypotheses.append(hypothesis)
                        probabilities.append(calculate_joint_probability(hypothesis, cov_inv))
        all_hypotheses.append(hypotheses)
        all_probabilities.append(probabilities)
    return all_hypotheses, all_probabilities

def find_max_associations(hypotheses, probabilities):
    """Find the most likely association for each track."""
    max_associations = []
    max_probs = []
    for track_hypotheses, track_probabilities in zip(hypotheses, probabilities):
        max_association = []
        max_prob = 0.0
        for hypothesis, prob in zip(track_hypotheses, track_probabilities):
            max_prob = max(max_prob, prob)
            max_association = max_association if max_prob > prob else hypothesis
        max_associations.append(max_association)
        max_probs.append(max_prob)
    return max_associations, max_probs
