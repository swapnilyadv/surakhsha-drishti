import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { NavigationContainer } from '@react-navigation/native';
import { LayoutDashboard, Video, Map, ShieldAlert, Settings } from 'lucide-react-native';
import { Colors } from '../theme/theme';

// Screens (To be created)
import { DashboardScreen } from '../screens/DashboardScreen';
import { LiveCCTVScreen } from '../screens/LiveCCTVScreen';
import { TacticalMapScreen } from '../screens/TacticalMapScreen';
import { IncidentsScreen } from '../screens/IncidentsScreen';
import { SettingsScreen } from '../screens/SettingsScreen';
import { LoginScreen } from '../screens/LoginScreen';
import { SplashScreen } from '../screens/SplashScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: Colors.surface,
          borderTopColor: Colors.border,
          height: 90,
          paddingBottom: 30,
        },
        tabBarActiveTintColor: Colors.primary,
        tabBarInactiveTintColor: Colors.textMuted,
        tabBarLabelStyle: {
          fontFamily: 'System',
          fontSize: 10,
          fontWeight: '600',
        },
        tabBarIcon: ({ color, size }) => {
          let icon;
          if (route.name === 'Dashboard') icon = <LayoutDashboard size={size} color={color} />;
          else if (route.name === 'Live') icon = <Video size={size} color={color} />;
          else if (route.name === 'Map') icon = <Map size={size} color={color} />;
          else if (route.name === 'Incidents') icon = <ShieldAlert size={size} color={color} />;
          else if (route.name === 'Settings') icon = <Settings size={size} color={color} />;
          return icon;
        },
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Live" component={LiveCCTVScreen} />
      <Tab.Screen name="Map" component={TacticalMapScreen} />
      <Tab.Screen name="Incidents" component={IncidentsScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export function AppNavigator() {
  const [showSplash, setShowSplash] = React.useState(true);
  const isAuthenticated = true; // Temporary

  if (showSplash) {
    return <SplashScreen onFinish={() => setShowSplash(false)} />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!isAuthenticated ? (
          <Stack.Screen name="Login" component={LoginScreen} />
        ) : (
          <Stack.Screen name="Main" component={MainTabs} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
