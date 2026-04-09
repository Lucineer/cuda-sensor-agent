//! # cuda-sensor-agent
//!
//! Perception agent that uses EquipmentRegistry to read sensors and report
//! confidence-weighted observations to the fleet.
//!
//! ```rust
//! use cuda_sensor_agent::SensorAgent;
//! use cuda_equipment::{EquipmentRegistry, SensorType, Confidence};
//!
//! let registry = EquipmentRegistry::new(1)
//!     .add_sensor("front_cam", SensorType::Camera, 1920)
//!     .add_sensor("thermistor", SensorType::Thermal, 12);
//!
//! let mut agent = SensorAgent::new(1, "perception", registry);
//! let obs = agent.observe("front_cam");
//! ```

pub use cuda_equipment::{Confidence, EquipmentRegistry, SensorType, ActuatorType,
    FleetMessage, MessageType, VesselId, Agent, TileGrid, Tile};

/// A sensor observation with confidence.
#[derive(Debug, Clone)]
pub struct Observation {
    pub sensor_name: String,
    pub sensor_type: SensorType,
    pub value: f64,
    pub confidence: Confidence,
    pub timestamp: u64,
    pub unit: String,
}

impl Observation {
    pub fn new(name: &str, stype: SensorType, value: f64, confidence: Confidence, unit: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self { sensor_name: name.to_string(), sensor_type: stype, value, confidence,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            unit: unit.to_string() }
    }
}

/// A perception agent that reads sensors and produces observations.
pub struct SensorAgent {
    id: VesselId,
    name: String,
    confidence: Confidence,
    registry: EquipmentRegistry,
    observations: Vec<Observation>,
    sensor_noise: f64,
    msg_count: (u64, u64),
}

impl SensorAgent {
    pub fn new(id: u64, name: &str, registry: EquipmentRegistry) -> Self {
        Self { id: VesselId(id), name: name.to_string(), confidence: Confidence::HALF,
            registry, observations: vec![], sensor_noise: 0.05, msg_count: (0, 0) }
    }

    pub fn with_noise(mut self, noise: f64) -> Self { self.sensor_noise = noise; self }

    /// Simulate reading a sensor (real implementation would read hardware).
    pub fn observe(&mut self, sensor_name: &str) -> Result<&Observation, String> {
        let sensor = self.registry.sensors.iter()
            .find(|s| s.name == sensor_name)
            .ok_or("sensor_not_found")?;
        
        let value = self.simulate_reading(&sensor.sensor_type, sensor.resolution);
        let conf = Confidence::new(sensor.confidence.value() - self.sensor_noise);
        
        let obs = Observation::new(sensor_name, sensor.sensor_type.clone(),
            value, conf, self.unit_for_type(&sensor.sensor_type));
        self.observations.push(obs);
        Ok(self.observations.last().unwrap())
    }

    /// Observe all sensors.
    pub fn observe_all(&mut self) -> Vec<&Observation> {
        let names: Vec<String> = self.registry.sensors.iter().map(|s| s.name.clone()).collect();
        names.into_iter().filter_map(|n| self.observe(&n).ok()).collect()
    }

    /// Get observations since a timestamp.
    pub fn observations_since(&self, ts: u64) -> Vec<&Observation> {
        self.observations.iter().filter(|o| o.timestamp > ts).collect()
    }

    /// Fuse multiple observations using Bayesian confidence.
    pub fn fuse(&self, observations: &[&Observation]) -> FusedObservation {
        if observations.is_empty() {
            return FusedObservation { value: 0.0, confidence: Confidence::ZERO,
                source_count: 0, sensor_types: vec![] };
        }
        let mut combined = observations[0].confidence;
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        let mut types = vec![];
        
        for obs in observations {
            combined = combined.combine(obs.confidence);
            let w = obs.confidence.value();
            weighted_sum += obs.value * w;
            weight_total += w;
            if !types.contains(&obs.sensor_type) { types.push(obs.sensor_type.clone()); }
        }
        
        let fused_value = if weight_total > 0.0 { weighted_sum / weight_total } else { 0.0 };
        FusedObservation { value: fused_value, confidence: combined,
            source_count: observations.len(), sensor_types: types }
    }

    fn simulate_reading(&self, stype: &SensorType, resolution: usize) -> f64 {
        // Simulation: produce a value based on sensor type
        // Real implementation reads hardware
        match stype {
            SensorType::Camera => (resolution * resolution / 1000) as f64, // "pixels" / 1000
            SensorType::Thermal => 22.0 + (self.observations.len() as f64 * 0.1).min(25.0), // temperature
            SensorType::Light => 50.0 + ((self.observations.len() * 7) % 100) as f64, // lux
            SensorType::Pressure => 1013.25 + (self.observations.len() as f64 * 0.1).min(5.0), // hPa
            SensorType::Humidity => 45.0 + ((self.observations.len() * 3) % 30) as f64, // %
            SensorType::Accelerometer => 9.81 + (self.observations.len() as f64 % 10) * 0.01,
            SensorType::Gyroscope => (self.observations.len() as f64 * 0.5) % 360.0,
            SensorType::Proximity => (10.0 - (self.observations.len() as f64 % 10) as f64).max(0.1),
            SensorType::Gps => 64.0 + (self.observations.len() as f64 * 0.001), // latitude offset
            _ => self.observations.len() as f64 * 0.1,
        }
    }

    fn unit_for_type(&self, stype: &SensorType) -> &str {
        match stype {
            SensorType::Camera => "mpx",
            SensorType::Thermal => "C",
            SensorType::Light => "lux",
            SensorType::Pressure => "hPa",
            SensorType::Humidity => "%",
            SensorType::Accelerometer => "m/s2",
            SensorType::Gyroscope => "deg/s",
            SensorType::Proximity => "m",
            SensorType::Gps => "deg",
            SensorType::Rf => "dBm",
            SensorType::Audio => "dB",
            SensorType::Magnetometer => "uT",
            _ => "units",
        }
    }
}

impl Agent for SensorAgent {
    fn id(&self) -> VesselId { self.id }
    fn name(&self) -> &str { &self.name }

    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage> {
        self.msg_count.1 += 1;
        match &msg.msg_type {
            MessageType::Ping => {
                self.msg_count.0 += 1;
                vec![msg.reply(MessageType::Pong)]
            }
            MessageType::CapabilityQuery => {
                self.msg_count.0 += 1;
                let caps = format!("sensors:{}", self.registry.sensors.iter()
                    .map(|s| s.name.as_str()).collect::<Vec<_>>().join(","));
                vec![msg.reply(MessageType::CapabilityResponse { capabilities: caps })]
            }
            MessageType::ConfidenceUpdate { confidence, .. } => {
                self.confidence = self.confidence.combine(*confidence);
                vec![]
            }
            _ => vec![],
        }
    }

    fn capabilities(&self) -> Vec<String> {
        self.registry.sensors.iter().map(|s| format!("sense:{}", s.name)).collect()
    }

    fn self_confidence(&self) -> Confidence { self.confidence }
}

/// Result of fusing multiple sensor observations.
#[derive(Debug, Clone)]
pub struct FusedObservation {
    pub value: f64,
    pub confidence: Confidence,
    pub source_count: usize,
    pub sensor_types: Vec<SensorType>,
}

/// Sensor health monitor — tracks sensor confidence degradation.
pub struct SensorHealth {
    readings_per_sensor: std::collections::HashMap<String, Vec<f64>>,
    max_history: usize,
}

impl SensorHealth {
    pub fn new(max_history: usize) -> Self {
        Self { readings_per_sensor: std::collections::HashMap::new(), max_history }
    }

    pub fn record(&mut self, sensor: &str, confidence: f64) {
        let history = self.readings_per_sensor.entry(sensor.to_string()).or_default();
        history.push(confidence);
        if history.len() > self.max_history { history.remove(0); }
    }

    pub fn trend(&self, sensor: &str) -> Option<f64> {
        let history = self.readings_per_sensor.get(sensor)?;
        if history.len() < 2 { return None; }
        let first = &history[..history.len()/2];
        let second = &history[history.len()/2..];
        let avg_first: f64 = first.iter().sum::<f64>() / first.len() as f64;
        let avg_second: f64 = second.iter().sum::<f64>() / second.len() as f64;
        Some(avg_second - avg_first)
    }

    pub fn is_degrading(&self, sensor: &str, threshold: f64) -> bool {
        self.trend(sensor).map_or(false, |t| t < -threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent() -> SensorAgent {
        let registry = EquipmentRegistry::new(1)
            .add_sensor("front_cam", SensorType::Camera, 1920)
            .add_sensor("thermistor", SensorType::Thermal, 12)
            .add_sensor("prox", SensorType::Proximity, 10);
        SensorAgent::new(1, "perception", registry)
    }

    #[test]
    fn test_observe_single() {
        let mut agent = make_agent();
        let obs = agent.observe("front_cam").unwrap();
        assert_eq!(obs.sensor_name, "front_cam");
        assert!(obs.value > 0.0);
    }

    #[test]
    fn test_observe_all() {
        let mut agent = make_agent();
        let all = agent.observe_all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_observe_missing() {
        let mut agent = make_agent();
        assert!(agent.observe("nonexistent").is_err());
    }

    #[test]
    fn test_fuse_observations() {
        let mut agent = make_agent();
        let all = agent.observe_all();
        let fused = agent.fuse(&all);
        assert!(fused.confidence.value() > 0.0);
        assert_eq!(fused.source_count, 3);
        assert_eq!(fused.sensor_types.len(), 3);
    }

    #[test]
    fn test_ping_pong() {
        let mut agent = make_agent();
        let ping = FleetMessage::new(VesselId(0), VesselId(1), MessageType::Ping);
        let responses = agent.receive(&ping);
        assert_eq!(responses.len(), 1);
    }

    #[test]
    fn test_capabilities() {
        let agent = make_agent();
        let caps = agent.capabilities();
        assert!(caps.iter().any(|c| c == "sense:front_cam"));
        assert!(caps.iter().any(|c| c == "sense:thermistor"));
    }

    #[test]
    fn test_sensor_health() {
        let mut health = SensorHealth::new(100);
        for i in 0..20 {
            health.record("cam", 0.9 - i as f64 * 0.03);
        }
        let trend = health.trend("cam").unwrap();
        assert!(trend < 0.0); // declining
        assert!(health.is_degrading("cam", 0.01));
    }

    #[test]
    fn test_confidence_update() {
        let mut agent = make_agent();
        let update = FleetMessage::new(VesselId(0), VesselId(1),
            MessageType::ConfidenceUpdate { topic: "health".to_string(), confidence: Confidence::LIKELY });
        agent.receive(&update);
        assert!(agent.self_confidence().value() > 0.5);
    }

    #[test]
    fn test_observations_since() {
        let mut agent = make_agent();
        agent.observe("thermistor").unwrap();
        let ts = agent.observations.last().unwrap().timestamp;
        agent.observe("prox").unwrap();
        let recent = agent.observations_since(ts);
        assert_eq!(recent.len(), 1);
    }
}
