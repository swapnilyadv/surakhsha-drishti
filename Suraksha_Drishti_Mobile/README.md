# Suraksha Drishti Mobile 📱

Tactical AI Surveillance Command Center for iOS & Android.

## Features
- **Live AI Monitoring**: Real-time stream processing with weapon & violence bounding boxes.
- **Tactical Map**: Geospatial incident tracking with threat-level indicators.
- **Biometric Security**: FaceID/Fingerprint authentication for law enforcement.
- **Forensic Logs**: Detailed incident history with evidence capture.
- **Cyberpunk UI**: Modern dark tactical interface designed for high-stakes monitoring.

## Tech Stack
- **React Native + Expo**
- **TypeScript**
- **Zustand** (State Management)
- **Reanimated** (Animations)
- **Socket.io** (Live Data Sync)
- **React Navigation** (Tactical Navigation)

## Getting Started

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Backend Configuration**:
   Update `BACKEND_URL` in `src/hooks/useSocket.ts` to your machine's IP address (e.g., `http://192.168.1.5:5001`).

3. **Run the App**:
   ```bash
   npx expo start
   ```

4. **Open on Device**:
   Scan the QR code with the Expo Go app.

## Project Structure
- `src/screens`: UI Screens (Dashboard, Map, CCTV, etc.)
- `src/store`: Zustand global state
- `src/theme`: Tactical design tokens
- `src/navigation`: App flow and routing
- `src/hooks`: WebSocket and API hooks
