# test_adapter.py
import cv2
from carla_adapter import CarlaAdapter

def main():
    adapter = CarlaAdapter(autopilot=True)
    adapter.start()

    print("✅ Adapter started. Press Q to quit.")

    try:
        while True:
            frame = adapter.get_frame()
            if frame is not None:
                cv2.imshow("Adapter Feed", frame)

            state = adapter.get_state()
            if state is not None:
                print(f"\rSpeed: {state.speed_mph:6.1f} mph | Throttle: {state.throttle:.2f} | Brake: {state.brake:.2f}", end="")

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        adapter.stop()
        cv2.destroyAllWindows()
        print("\nCleaned up.")

if __name__ == "__main__":
    main()
