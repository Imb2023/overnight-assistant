# Modular Ceiling Safety Node — Hardware Direction

Updated: 2026-10-08

## Goal

Explore a low-cost, privacy-first room safety node that can be mounted high on a wall or ceiling and observe a room locally. The project should remain modular: camera/phone, compute, motion hardware, controller, sensors, communications, and alerting must be replaceable independently.

This is an experimental assistive-safety project, not a validated medical device.

## Build philosophy

**Use hardware already available first. Upgrade a module only when testing demonstrates a real limitation.**

The 2024 prototype proved the basic idea. V2 does not need to preserve its old software architecture. Development should move toward the goal through small, measurable experiments and record what works in Git.

## Current first prototype

The current available camera is a **Logitech QuickCam Pro 9000** USB webcam. Use it as the first reference camera rather than purchasing an iPhone, depth camera, or new webcam before those capabilities are shown to be necessary.

Initial architecture:

`Logitech QuickCam Pro 9000 -> local compute (PC/Raspberry Pi) -> local vision -> BLE command -> ESP32 -> optional pan/tilt servos`

Start with the camera fixed. Motorized movement is an optional later module, not a requirement for the first vision experiment.

The QuickCam Pro 9000 is a reference implementation, not a permanent dependency. Before relying on model-specific capabilities, confirm the exact attached device and supported capture modes on the target computer (for Linux, for example, enumerate the USB/V4L2 device and its actual formats). Do not design V2 around an assumed resolution or frame rate.

A future iPhone remains a possible self-contained camera/compute upgrade, especially if testing shows value from newer cameras, on-device acceleration, depth-capable models, or battery backup. It should only replace the current camera/compute modules when it provides a demonstrated benefit.

## Modular boundaries

### Sensor module

Initial reference: Logitech QuickCam Pro 9000 USB webcam already available.

Possible replacements: another UVC webcam, repurposed phone, depth camera, thermal array, mmWave radar, or combinations of sensors.

The sensor interface should not dictate the fall/event reasoning architecture.

### Compute / vision module

Initial reference: an available PC or Raspberry Pi capable of accepting the USB camera.

Responsibilities may include camera capture, person/pose inference, event reasoning, and high-level movement requests. Keep vision software independent of the particular motor controller and camera model where practical.

### Motion controller

Initial option: ESP32 development board.

Responsibilities: BLE/local command reception, servo control, limits, centering/home behavior, and watchdog/fail-safe behavior. Keep the motor protocol simple enough that the camera/vision compute module can later be replaced.

### Pan/tilt mechanism

Use two independently replaceable servos and commodity brackets rather than a complete multi-axis robot arm unless testing proves additional axes are needed.

Do not buy motion hardware merely because tracking sounds useful. First measure whether a fixed high-mounted camera provides adequate room coverage.

### Communications

BLE is a candidate for local node/controller communication and nearby-node status exchange. Do not assume BLE alone is sufficient for final emergency delivery. Nodes should be able to detect loss of peer/controller heartbeat rather than interpreting silence as `OK`.

Possible future node messages:

- `NODE_ALIVE`
- `PERSON_OK`
- `PERSON_PRESENT`
- `POSSIBLE_FALL`
- `PERSON_DOWN`
- `NODE_FAULT`

These messages should contain state/telemetry rather than room video.

## Reference parts and observed prices

Prices and stock are snapshots, not requirements. Verify before purchasing. Equivalent parts are acceptable when the electrical/mechanical interface is compatible.

### Camera — current cost: $0

- Logitech QuickCam Pro 9000 — already available; use as the initial USB camera.
- Purchase cost for the current prototype: **$0**.
- Replacement rule: use another supported USB/UVC camera or sensor if the existing camera proves inadequate.

### Controller

- Elegoo ESP-32 Development Board, 3-pack — Micro Center Dallas observed price: **$19.99** (~$6.67 per board).
- Substitute: any supported ESP32-class board with BLE and enough PWM/control capability for the chosen motor driver/servos.

### Servos

Micro Center Dallas examples observed during research:

- Hiwonder HPS-2018 20kg full-metal-gear servo — **$14.99 each**.
- Hiwonder HPS-2027 20kg high-torque digital full-metal-gear servo — **$15.99 each**.
- Hiwonder HX-20L 20kg serial-bus servo with feedback — **$16.99 each**.

For an initial two-axis mechanism, two servos are required. Do not choose solely by torque rating: verify voltage, current, control protocol, travel, feedback needs, bracket/horn compatibility, noise, duty cycle, and the actual mounted mass.

### Mechanical brackets

Hiwonder commodity servo-frame examples at Micro Center were observed around **$1.99 each**, including U-shaped frames and servo horns. These are examples only. A generic metal pan/tilt bracket, 3D-printed mount, or custom bracket is acceptable.

### Power

Servo power should be designed separately from the ESP32/camera-compute supply as appropriate. Micro Center examples included Hiwonder robot power adapters around **$8.99–$14.99**, but voltage/current must match the selected servos. Never select a supply merely because it is marketed for robotics.

### Future depth option

Hiwonder Aurora930 Pro structured-light depth camera was observed at Micro Center Dallas for **$149.99**. This is not required for the first prototype; it is recorded as an example of a future replaceable depth-sensing module.

## Approximate costs

### Stage 1 — vision experiment

Use the existing Logitech camera and available PC/Raspberry Pi.

**Incremental camera cost: $0.**

This stage should answer whether we can reliably acquire the room view and run useful local person/pose observations before buying robotics hardware.

### Stage 2 — optional motion/controller experiment

Using the observed reference prices:

- ESP32 share from 3-pack: ~$6.67
- 2 x HPS-2018 servos: ~$29.98
- several basic brackets/horns: roughly $4–$10 depending on geometry
- appropriate power supply: roughly $9–$15 reference range

This puts the optional core experimental controller/motion hardware roughly in the **$50–$65** range before ceiling hardware, enclosure, cabling, taxes, and any custom mount. This is a planning estimate, not a quoted build price.

## Substitution rules

If a listed product disappears, preserve the interface rather than the brand:

1. **Camera/sensor:** supported camera, phone, depth, radar, thermal, or other local sensor with a documented adapter.
2. **Compute:** PC, Raspberry Pi, phone, or future edge computer capable of the selected inference workload.
3. **Controller:** BLE-capable microcontroller; ESP32-class is the initial reference.
4. **Actuator:** standard PWM servo or documented serial/bus servo with adequate torque and known protocol.
5. **Mount:** mechanically compatible pan/tilt structure rated for the mounted device plus safety margin.
6. **Power:** sized from measured peak/stall loads and manufacturer requirements.
7. **Communication:** abstract messages; do not hard-code the project around one radio or vendor SDK.

## Safety requirements for overhead mounting

A camera or motor assembly above a person creates a physical hazard. Any ceiling prototype must eventually include a primary structural mount plus an independent safety tether/secondary retention. Servo torque, fasteners, bracket fatigue, cable strain relief, and failure position must be considered before placing a prototype over an occupied sleeping area.

For early software/motion testing, operate the assembly on a bench or low stand rather than over a person.

## Revised small-step development path

1. Connect the existing Logitech QuickCam Pro 9000 to the intended PC/Raspberry Pi.
2. Record the exact detected USB/V4L2 identity and supported capture modes.
3. Establish reliable local camera capture.
4. Add local person/pose observation while the camera remains fixed.
5. Test several useful mounting positions and measure room coverage/blind spots.
6. Decide from those measurements whether pan/tilt is actually necessary.
7. Only if movement is justified: bench-test one ESP32 and one servo and establish BLE command/heartbeat communication.
8. Add the second servo, limits, and home position.
9. Use a lightweight dummy camera mass for mechanical testing before mounting valuable hardware.
10. Connect local person/pose tracking to slow movement commands only after the fixed-camera and mechanical experiments are independently working.
11. Later test two room nodes exchanging state/heartbeat messages without exchanging video.

## Design principle

Overnight Assistant should evolve as replaceable modules rather than a single hardware kit. Current retail products are reference implementations, not dependencies. The source of truth should record interfaces, measurements, test results, failures, and known-good combinations so another builder can substitute available parts later.
