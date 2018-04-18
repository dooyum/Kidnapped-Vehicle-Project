/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static const double EPS = 0.0001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 12;
  default_random_engine gen;

  // Create normal (Gaussian) distributions for x, y and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned int i = 0; i < num_particles; i++) {
    Particle particle = Particle();
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  double const std_x = std_pos[0];
  double const std_y = std_pos[1];
  double const std_theta = std_pos[2];

  for (Particle &particle : particles) {
    double const x = particle.x;
    double const y = particle.y;
    double const theta = particle.theta;
    double x_pred;
    double y_pred;
    double yaw_pred;

    double const yaw_rate_dt = yaw_rate * delta_t;
    yaw_pred = theta + yaw_rate_dt;

    if (fabs(yaw_rate) > EPS) {
      double const vel_yaw_rate_ratio = velocity/yaw_rate;
      x_pred = x + vel_yaw_rate_ratio * (sin(theta + yaw_rate_dt) - sin(theta));
      y_pred = y + vel_yaw_rate_ratio * (cos(theta) - cos(theta + yaw_rate_dt));
    } else {
      x_pred = x + velocity * delta_t * cos(theta);
      y_pred = y + velocity * delta_t * sin(theta);
    }


    // Add noise for predicted x, y and yaw using normal (Gaussian) distributions.
    normal_distribution<double> dist_x(x_pred, std_x);
    normal_distribution<double> dist_y(y_pred, std_y);
    normal_distribution<double> dist_theta(yaw_pred, std_theta);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> close_landmarks, std::vector<LandmarkObs>&observations) {
  // Return early if there are no landmarks to compare to.
  if (close_landmarks.empty()) return;
  for (LandmarkObs &observation : observations) {
    double shortest_distance = __DBL_MAX__;
    LandmarkObs closest_landmark;
    for (LandmarkObs landmark : close_landmarks) {
      double distance = dist(observation.x, observation.y, landmark.x, landmark.y);
      if (distance < shortest_distance) {
        closest_landmark = landmark;
        shortest_distance = distance;
      }
    }
    observation.id = closest_landmark.id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  double const std_x = std_landmark[0];
  double const std_y = std_landmark[1];

  for (unsigned int i = 0; i < particles.size(); i++) {
    Particle particle = particles[i];
    // Convert vehicle observations from vehicle coordinates to map cooordinates.
    std::vector<LandmarkObs> transformed_observations;
    for (LandmarkObs observation : observations) {
      double const x_map= particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
      double const y_map= particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
      LandmarkObs map_observation = LandmarkObs();
      map_observation.id = observation.id;
      map_observation.x = x_map;
      map_observation.y = y_map;
      transformed_observations.push_back(map_observation);
    }

    // Get the possible landmarks from the map by filtering for those that are within the sensor range.
    std::vector<LandmarkObs> possible_landmarks;
    for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
      LandmarkObs possible_landmark = LandmarkObs();
      possible_landmark.id = landmark.id_i;
      possible_landmark.x = landmark.x_f;
      possible_landmark.y = landmark.y_f;
      double distance = dist(particle.x, particle.y, possible_landmark.x, possible_landmark.y);
      if (distance < sensor_range * 2) {
        possible_landmarks.push_back(possible_landmark);
      }
    }

    dataAssociation(possible_landmarks, transformed_observations);

    // Calculate normalization term.
    double const gaussian_normalizer = (1/(2 * M_PI * std_x * std_y));
    // Penalize particle with low weights if there's no close landmark found.
    double weight = possible_landmarks.empty() ? EPS : 1.0;

    // Update particle weight.
    for (LandmarkObs observation : transformed_observations) {
      for (LandmarkObs landmark : possible_landmarks) {
        // Add weights for closest identified landmarks.
        if (observation.id == landmark.id) {
          double const exponent = pow(observation.x - landmark.x, 2) / (2 * pow(std_x,2)) +
            pow(observation.y - landmark.y, 2) / 2 * pow(std_y, 2);
          weight *= gaussian_normalizer * exp(-exponent);
        }
      }
    }
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  double const max_particle_weight = *std::max_element(weights.begin(), weights.end());
  double beta = 0.0;
  default_random_engine gen;
  discrete_distribution<double> distribution(num_particles, 1);
  std::vector<Particle> sampled_particles;

  for (unsigned int i = 0; i < num_particles; i++) {
    beta += distribution(gen) * 2.0 * max_particle_weight;
    unsigned int index = 0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    sampled_particles.push_back(particles[index]);
  }
  particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
