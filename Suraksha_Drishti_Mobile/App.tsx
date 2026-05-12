import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AppNavigator } from './src/navigation/AppNavigator';
import { useSocket } from './src/hooks/useSocket';
import { AlertBanner } from './src/components/AlertBanner';

export default function App() {
  // Initialize WebSocket connection
  useSocket();

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <StatusBar style="light" />
        <AppNavigator />
        <AlertBanner />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
