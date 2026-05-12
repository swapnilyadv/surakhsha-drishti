import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Animated, { SlideInDown, SlideOutDown } from 'react-native-reanimated';
import { ShieldAlert, X } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { useAppStore } from '../store/useAppStore';

export function AlertBanner() {
  const { detections, setThreatLevel } = useAppStore();
  
  const activeAlert = Object.values(detections).find(d => d.weapon || d.violence);

  if (!activeAlert) return null;

  return (
    <Animated.View 
      entering={SlideInDown} 
      exiting={SlideOutDown}
      style={styles.container}
    >
      <View style={styles.iconContainer}>
        <ShieldAlert color="#fff" size={24} />
      </View>
      <View style={styles.content}>
        <Text style={styles.title}>CRITICAL THREAT DETECTED</Text>
        <Text style={styles.subtitle}>
          {activeAlert.weapon ? `WEAPON: ${activeAlert.weapon_type}` : 'VIOLENCE DETECTED'} IN {activeAlert.cam_id}
        </Text>
      </View>
      <TouchableOpacity 
        style={styles.closeBtn}
        onPress={() => setThreatLevel('SAFE')}
      >
        <X size={20} color="rgba(255,255,255,0.5)" />
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 100,
    left: 20,
    right: 20,
    backgroundColor: Colors.destructive,
    borderRadius: 16,
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    shadowColor: Colors.destructive,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
    zIndex: 1000,
  },
  iconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    flex: 1,
    marginLeft: 15,
  },
  title: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 10,
    fontWeight: '700',
    marginTop: 2,
  },
  closeBtn: {
    padding: 5,
  },
});
