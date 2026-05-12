import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withTiming, 
  withDelay, 
  withRepeat, 
  withSequence,
  Easing
} from 'react-native-reanimated';
import { Shield } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { LinearGradient } from 'expo-linear-gradient';

export function SplashScreen({ onFinish }: { onFinish: () => void }) {
  const opacity = useSharedValue(0);
  const scale = useSharedValue(0.8);
  const scanLinePos = useSharedValue(-100);

  useEffect(() => {
    opacity.value = withTiming(1, { duration: 1000 });
    scale.value = withTiming(1, { duration: 1000, easing: Easing.out(Easing.back(1.5)) });
    
    scanLinePos.value = withRepeat(
      withTiming(300, { duration: 2000, easing: Easing.linear }),
      -1,
      false
    );

    const timer = setTimeout(onFinish, 3000);
    return () => clearTimeout(timer);
  }, []);

  const animatedLogoStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ scale: scale.value }],
  }));

  const scanLineStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: scanLinePos.value }],
  }));

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#020617', '#0f172a']} style={StyleSheet.absoluteFill} />
      
      <Animated.View style={[styles.logoContainer, animatedLogoStyle]}>
        <View style={styles.hex}>
          <Shield size={60} color={Colors.primary} />
          <Animated.View style={[styles.scanLine, scanLineStyle]} />
        </View>
        <Text style={styles.title}>SURAKSHA DRISHTI</Text>
        <Text style={styles.subtitle}>AI SURVEILLANCE CORE v2.4</Text>
      </Animated.View>

      <View style={styles.loaderContainer}>
        <Text style={styles.loadingText}>INITIALIZING TACTICAL NODE...</Text>
        <View style={styles.progressBar}>
          <Animated.View style={styles.progressFill} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  logoContainer: { alignItems: 'center' },
  hex: { 
    width: 120, 
    height: 120, 
    backgroundColor: Colors.surface, 
    justifyContent: 'center', 
    alignItems: 'center', 
    borderRadius: 24,
    borderWidth: 2,
    borderColor: Colors.primary,
    overflow: 'hidden',
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    marginBottom: 20
  },
  scanLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: Colors.primary,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 10,
  },
  title: { color: '#fff', fontSize: 24, fontWeight: '900', letterSpacing: 4 },
  subtitle: { color: Colors.primary, fontSize: 10, fontWeight: 'bold', letterSpacing: 2, marginTop: 10 },
  
  loaderContainer: { position: 'absolute', bottom: 80, width: '60%' },
  loadingText: { color: Colors.textMuted, fontSize: 8, fontWeight: 'bold', textAlign: 'center', marginBottom: 10, letterSpacing: 1 },
  progressBar: { height: 2, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 1 },
  progressFill: { width: '100%', height: '100%', backgroundColor: Colors.primary },
});
