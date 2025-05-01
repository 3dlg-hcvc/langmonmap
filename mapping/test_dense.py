def project_dense(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        
        if self.fusion_type == FusionType.EMA:
            indices = confidences_mapped.indices()
            indices_obstacle = obstacle_mapped.indices()
            confs_new = confidences_mapped.values().data.squeeze()
            confs_old = self.confidence_map[indices[0], indices[1]]

            confs_old_obs = self.confidence_map[indices_obstacle[0], indices_obstacle[1]]

            confidence_denominator = confs_new + confs_old
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.updated_mask[indices[0], indices[1]] = True

            self.feature_map[indices[0], indices[1]] = self.feature_map[indices[0], indices[1]] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)

            self.confidence_map[indices[0], indices[1]] = confidence_denominator

            # we also need to update the checked confidence
            confs_old_checked = self.checked_conf_map[indices[0], indices[1]]
            confidence_denominator_checked = confs_new + confs_old_checked
            self.checked_conf_map[indices[0], indices[1]] = confidence_denominator_checked

            # Obstacle Map update
            confs_new = obstcl_confidence_mapped.values().data.squeeze()
            confidence_denominator = confs_new + confs_old_obs
            weight_1 = torch.nan_to_num(confs_old_obs / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.obstacle_map[indices_obstacle[0], indices_obstacle[1]] = self.obstacle_map[
                                                                              indices_obstacle[0], indices_obstacle[
                                                                                  1]] * weight_1 + \
                                                                          obstacle_mapped.values().data.squeeze() * weight_2

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    self.occluded_map[obs[0], obs[1]] = True
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)


            self.fully_explored_map = (np.nan_to_num(1.0 / self.confidence_map.cpu().numpy())
                                       < self.fully_explored_threshold)

            self.checked_map = (np.nan_to_num(1.0 / self.checked_conf_map.cpu().numpy())
                                < self.checked_map_threshold)
