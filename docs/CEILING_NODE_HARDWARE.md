# Modular Ceiling Safety Node — Hardware Direction

Updated: 2026-10-08

## Goal

Explore a low-cost, privacy-first room safety node that can be mounted high on a wall or ceiling and observe a room locally. The project should remain modular: camera/phone, motion hardware, controller, sensors, communications, and alerting must be replaceable independently.

This is an experimental assistive-safety project, not a validated medical device.

## Current prototype direction

Start with the smallest useful experiment rather than a complete robot arm:

`old iPhone or other camera/compute device -> local vision -> BLE command -> ESP32 -> pan/tilt servos`

A fixed mount should be tested first. Add pan/tilt only if fixed wide-angle coverage leaves important blind spots.

The phone is not a permanent requirement. It is one convenient prototype sensor/compute module. Future nodes may use another phone, Raspberry Pi, dedicated camera/depth module, thermal sensor, mmWave radar, or combinations of these.

## Modular boundaries

### Sensor / compute module

Initial option: repurposed iPhone.

Responsibilities may include local camera capture, person/pose inference, event reasoning, and issuing high-level movement requests. Do not make the mechanical controller dependent on a particular phone model.

### Motion controller

Initial option: ESP32 development board.

Responsibilities: BLE/local command reception, servo control, limits, centering/home behavior, and watchdog/fail-safe behavior. Keep the motor protocol simple enough that the phone/vision module can later be replaced.

### Pan/tilt mechanism

Use two independently replaceable servos and commodity brackets rather than a complete multi-axis robot arm unless testing proves additional axes are needed.

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

Servo power should be designed separately from the ESP32/phone supply as appropriate. Micro Center examples included Hiwonder robot power adapters around **$8.99–$14.99**, but voltage/current must match the selected servos. Never select a supply merely because it is marketed for robotics.

### Future depth option

Hiwonder Aurora930 Pro structured-light depth camera was observed at Micro Center Dallas for **$149.99**. This is not required for the first prototype; it is recorded as an example of a future replaceable depth-sensing module.

## Approximate initial motion-controller cost

Using the observed reference prices:

- ESP32 share from 3-pack: ~$6.67
- 2 x HPS-2018 servos: ~$29.98
- several basic brackets/horns: roughly $4–$10 depending on geometry
- appropriate power supply: roughly $9–$15 reference range

This puts the core experimental controller/motion hardware roughly in the **$50–$65** range before the phone, ceiling hardware, enclosure, cabling, taxes, and any custom mount. This is a planning estimate, not a quoted build price.

## Substitution rules

If a listed product disappears, preserve the interface rather than the brand:

1. **Controller:** BLE-capable microcontroller; ESP32-class is the initial reference.
2. **Actuator:** standard PWM servo or documented serial/bus servo with adequate torque and known protocol.
3. **Mount:** mechanically compatible pan/tilt structure rated for the mounted device plus safety margin.
4. **Sensor:** replaceable camera/phone/depth/radar/thermal module.
5. **Power:** sized from measured peak/stall loads and manufacturer requirements.
6. **Communication:** abstract messages; do not hard-code the project around one radio or vendor SDK.

## Safety requirements for overhead mounting

A phone or motor assembly above a person creates a physical hazard. Any ceiling prototype must eventually include a primary structural mount plus an independent safety tether/secondary retention. Servo torque, fasteners, bracket fatigue, cable strain relief, and failure position must be considered before placing a prototype over an occupied sleeping area.

For early software/motion testing, operate the assembly on a bench or low stand rather than over a person.

## Small-step development path

1. Bench-test one ESP32 and one servo.
2. Establish BLE command/heartbeat communication.
3. Add second servo and implement pan/tilt limits and home position.
4. Attach a lightweight dummy mass before attaching a phone.
5. Test a fixed phone/camera view and determine whether motorized tracking is actually needed.
6. If needed, mount phone on the pan/tilt assembly and test slow commanded positioning.
7. Only then connect local person/pose tracking to movement commands.
8. Later test two nodes exchanging state/heartbeat messages without exchanging video.

## Design principle

Overnight Assistant should evolve as replaceable modules rather than a single hardware kit. Current retail products are reference implementations, not dependencies. The source of truth should record interfaces, measurements, test results, and known-good combinations so another builder can substitute available parts later.
