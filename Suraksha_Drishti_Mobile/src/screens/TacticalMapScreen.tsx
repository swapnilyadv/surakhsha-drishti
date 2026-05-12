import React from 'react';
import { View, Text, StyleSheet, Dimensions, TouchableOpacity } from 'react-native';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';
import { Colors } from '../theme/theme';
import { useAppStore } from '../store/useAppStore';
import { Crosshair, ShieldAlert, Navigation } from 'lucide-react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width, height } = Dimensions.get('window');

const INITIAL_REGION = {
  latitude: 18.5204, // Pune
  longitude: 73.8567,
  latitudeDelta: 0.05,
  longitudeDelta: 0.05,
};

const CAM_LOCATIONS = [
  { id: 'CAM-001', name: 'Main Entrance', lat: 18.5204, lng: 73.8567 },
  { id: 'CAM-002', name: 'Parking Lot A', lat: 18.5250, lng: 73.8600 },
  { id: 'CAM-003', name: 'North Corridor', lat: 18.5150, lng: 73.8500 },
  { id: 'CAM-004', name: 'South Gate', lat: 18.5300, lng: 73.8700 },
];

export function TacticalMapScreen() {
  const { detections } = useAppStore();

  return (
    <View style={styles.container}>
      <MapView
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        initialRegion={INITIAL_REGION}
        customMapStyle={darkMapStyle}
      >
        {CAM_LOCATIONS.map((loc) => {
          const det = detections[loc.id];
          const isCritical = det?.weapon || det?.violence || det?.density_level === 'CRITICAL';
          const isWarning = det?.density_level === 'HIGH';
          
          let color = Colors.success;
          if (isCritical) color = Colors.destructive;
          else if (isWarning) color = Colors.warning;

          return (
            <Marker
              key={loc.id}
              coordinate={{ latitude: loc.lat, longitude: loc.lng }}
              title={loc.name}
              description={`${loc.id} | AI Node Active`}
            >
              <View style={[styles.markerContainer, { borderColor: color }]}>
                <View style={[styles.markerCore, { backgroundColor: color }]} />
                {(isCritical || isWarning) && (
                  <View style={[styles.markerPulse, { backgroundColor: color }]} />
                )}
              </View>
            </Marker>
          );
        })}
      </MapView>

      <SafeAreaView style={styles.overlay}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>TACTICAL OVERLAY</Text>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>ENCRYPTED GPS</Text>
          </View>
        </View>

        <View style={styles.controls}>
          <TouchableOpacity style={styles.controlBtn}>
            <Crosshair size={24} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.controlBtn}>
            <Navigation size={24} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity style={[styles.controlBtn, { backgroundColor: Colors.destructive }]}>
            <ShieldAlert size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        <View style={styles.legend}>
          <LegendItem color={Colors.destructive} label="CRITICAL" />
          <LegendItem color={Colors.warning} label="WARNING" />
          <LegendItem color={Colors.success} label="SECURE" />
        </View>
      </SafeAreaView>
    </View>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: color }]} />
      <Text style={styles.legendLabel}>{label}</Text>
    </View>
  );
}

const darkMapStyle = [
  { "elementType": "geometry", "stylers": [{ "color": "#020617" }] },
  { "elementType": "labels.text.fill", "stylers": [{ "color": "#475569" }] },
  { "elementType": "labels.text.stroke", "stylers": [{ "color": "#020617" }] },
  { "featureType": "administrative", "elementType": "geometry.stroke", "stylers": [{ "color": "#1e293b" }] },
  { "featureType": "road", "elementType": "geometry", "stylers": [{ "color": "#0f172a" }] },
  { "featureType": "road", "elementType": "geometry.stroke", "stylers": [{ "color": "#1e293b" }] },
  { "featureType": "water", "elementType": "geometry", "stylers": [{ "color": "#001021" }] }
];

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  map: { width: width, height: height },
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'box-none' },
  header: { padding: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  headerTitle: { color: Colors.primary, fontSize: 16, fontWeight: '900', letterSpacing: 2 },
  badge: { backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 4, borderWidth: 1, borderColor: Colors.primary + '40' },
  badgeText: { color: Colors.primary, fontSize: 8, fontWeight: 'bold' },
  
  controls: { position: 'absolute', right: 20, bottom: 120, gap: 15 },
  controlBtn: { width: 50, height: 50, borderRadius: 25, backgroundColor: 'rgba(15,23,42,0.9)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: Colors.border, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.5, shadowRadius: 10 },
  
  legend: { position: 'absolute', left: 20, bottom: 120, backgroundColor: 'rgba(15,23,42,0.9)', padding: 12, borderRadius: 12, borderWidth: 1, borderColor: Colors.border, gap: 8 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  legendDot: { width: 8, height: 8, borderRadius: 4 },
  legendLabel: { color: Colors.textMuted, fontSize: 10, fontWeight: 'bold' },

  markerContainer: { width: 24, height: 24, borderRadius: 12, borderWidth: 2, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center' },
  markerCore: { width: 10, height: 10, borderRadius: 5 },
  markerPulse: { position: 'absolute', width: 40, height: 40, borderRadius: 20, opacity: 0.3, zIndex: -1 },
});
