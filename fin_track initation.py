def group_measurements_into_tracks(measurements):
    """Group measurements into tracks."""
    tracks = []
    used_indices = set()
    
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        track_range = np.linalg.norm(np.array([x_base, y_base, z_base]))
        
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            
            range_diff = np.abs(np.linalg.norm(np.array([x, y, z])) - track_range)
            if range_diff <= 1.0:  # Adjust the threshold as needed
                track.append((x, y, z, mt))
                used_indices.add(j)
        
        tracks.append(track)
    
    return tracks

def main():
    """Main processing loop."""
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84_2.csv'

    try:
        measurements = read_measurements_from_csv(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not measurements:
        print("No measurements found in the CSV file.")
        return

    tracks = group_measurements_into_tracks(measurements)
    cov_inv = np.linalg.inv(np.eye(state_dim))  # Example covariance inverse matrix

    predicted_states = []
    updated_states = []

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []

        for i, (x, y, z, mt) in enumerate(track_group):
            if i == 0:
                # Initialize filter state with the first measurement
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            else:
                kalman_filter.predict_step(mt)
                kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)
                reports.append(np.array([x, y, z]))

            # predicted_states.append((kalman_filter.Meas_Time, np.linalg.norm(kalman_filter.Sf[:3]), *cart2sph(*kalman_filter.Sf[:3])))

        # Perform clustering, hypothesis generation, and association
        hypotheses, probabilities = perform_clustering_hypothesis_association(track_group, reports, cov_inv)

        # # Print hypotheses and their respective probabilities
        # for hypothesis, probability in zip(hypotheses, probabilities):
        #     print(f"Hypothesis: {hypothesis}, Probability: {probability}")

        # Find the most likely association for each report
        max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

        for report_idx, track_idx in enumerate(max_associations):
            if track_idx != -1:
                kalman_filter.update_step(reports[report_idx])
                updated_states.append((kalman_filter.Meas_Time, np.linalg.norm(kalman_filter.Sf[:3]), *cart2sph(*kalman_filter.Sf[:3])))

        # Print associated measurements to track
        for i, (report_idx, track_idx) in enumerate(zip(max_associations, track_group)):
            if report_idx != -1:
                print(f"Report {i+1} associated with Track {track_idx}")

        # Print the most likely associated measurements for each target
        for i, (report_idx, prob) in enumerate(zip(max_associations, max_probs)):
            if report_idx != -1:
                print(f"Most likely association for report {i+1}: Track {report_idx} with probability {prob}")

    # Plotting all data together
    plot_track_data(predicted_states, updated_states)

if __name__ == "__main__":
    main()
