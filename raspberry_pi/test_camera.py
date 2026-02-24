#!/usr/bin/env python3
"""
Quick Test Script for Raspberry Pi Camera
Tests if Zebronics camera is working before running main GUI
"""

import cv2
import sys

print("=" * 60)
print("Raspberry Pi Camera Test")
print("=" * 60)

# Test camera index 0
print("\n1. Testing camera at /dev/video0...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ Camera 0 is working!")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print("   Press 'q' to quit the test window")
        
        # Show camera feed
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera test successful!")
        sys.exit(0)
    else:
        print("❌ Camera 0 opened but cannot read frames")
        cap.release()
else:
    print("❌ Cannot open camera at index 0")

# Test camera index 1
print("\n2. Testing camera at /dev/video1...")
cap = cv2.VideoCapture(1)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ Camera 1 is working!")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print("   Press 'q' to quit the test window")
        
        # Show camera feed
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera test successful!")
        sys.exit(0)
    else:
        print("❌ Camera 1 opened but cannot read frames")
        cap.release()
else:
    print("❌ Cannot open camera at index 1")

# No camera found
print("\n" + "=" * 60)
print("❌ ERROR: No working camera found!")
print("=" * 60)
print("\nTroubleshooting steps:")
print("1. Check USB cable connection")
print("2. Try different USB port")
print("3. Run: lsusb (should show camera device)")
print("4. Run: ls /dev/video* (should show video devices)")
print("5. Check camera permissions")
print("=" * 60)
sys.exit(1)
