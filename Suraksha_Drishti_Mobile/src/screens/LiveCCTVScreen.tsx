import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, Image, TouchableOpacity, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Maximize2, ShieldAlert, Crosshair } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { useAppStore } from '../store/useAppStore';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

const CAMERAS = [
  { id: 'CAM-001', name: 'Main Entrance', source: 'http://127.0.0.1:5001/api/stream/CAM-001' },
  { id: 'CAM-002', name: 'Parking Lot A', source: 'http://127.0.0.1:5001/api/stream/CAM-002' },
  { id: 'CAM-003', name: 'North Corridor', source: 'http://127.0.0.1:5001/api/stream/CAM-003' },
  { id: 'CAM-004', name: 'South Gate', source: 'http://127.0.0.1:5001/api/stream/CAM-004' },
];

export function LiveCCTVScreen() {
  const { detections } = useAppStore();
  const [selectedCam, setSelectedCam] = useState<string | null>(null);

  const renderItem = ({ item }: { item: typeof CAMERAS[0] }) => {
    const det = detections[item.id];
    const hasAlert = det?.violence || det?.weapon;

    return (
      <View style={styles.cameraCard}>
        <View style={styles.cameraHeader}>
          <Text style={styles.cameraName}>{item.name}</Text>
          <View style={styles.badgeContainer}>
            <View style={[styles.dot, { backgroundColor: Colors.success }]} />
            <Text style={styles.badgeText}>LIVE</Text>
          </View>
        </View>

        <View style={styles.streamContainer}>
          <Image 
            source={{ uri: item.source }} 
            style={styles.stream} 
            resizeMode="cover"
          />
          
          {/* AI Overlay Simulation */}
          {hasAlert && (
            <View style={styles.alertOverlay}>
              <View style={styles.boundingBox} />
              <View style={styles.alertTag}>
                <ShieldAlert size={12} color="#fff" />
                <Text style={styles.alertTagText}>
                  {det?.weapon ? `WEAPON: ${det.weapon_type}` : 'VIOLENCE'}
                </Text>
              </View>
            </View>
          )}

          <TouchableOpacity style={styles.expandButton} onPress={() => setSelectedCam(item.id)}>
            <Maximize2 size={20} color="#fff" />
          </TouchableOpacity>
          
          <View style={styles.hudOverlay}>
            <Text style={styles.hudText}>{item.id} | FPS: 15 | AI: ACTIVE</Text>
          </View>
        </View>

        {det && (
          <View style={styles.cameraFooter}>
            <View style={styles.footerStat}>
              <Text style={styles.statLabel}>PEOPLE</Text>
              <Text style={styles.statValue}>{det.crowd_count || 0}</Text>
            </View>
            <View style={styles.footerStat}>
              <Text style={styles.statLabel}>CONFIDENCE</Text>
              <Text style={styles.statValue}>{Math.round(det.confidence * 100)}%</Text>
            </View>
            <View style={styles.footerStat}>
              <Text style={styles.statLabel}>DENSITY</Text>
              <Text style={[styles.statValue, { color: det.density_level === 'CRITICAL' ? Colors.destructive : Colors.text }]}>
                {det.density_level || 'LOW'}
              </Text>
            </View>
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={[Colors.surface, Colors.background]} style={StyleSheet.absoluteFill} />
      
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>TACTICAL STREAMS</Text>
          <Text style={styles.headerSubtitle}>MULTI-CAMERA SURVEILLANCE NODE</Text>
        </View>
        <TouchableOpacity style={styles.gridButton}>
          <Crosshair size={24} color={Colors.primary} />
        </TouchableOpacity>
      </View>

      <FlatList
        data={CAMERAS}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContent}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { 
    flexDirection: 'row', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    paddingHorizontal: 20, 
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border
  },
  headerTitle: { color: Colors.primary, fontSize: 16, fontWeight: '900', letterSpacing: 1.5 },
  headerSubtitle: { color: Colors.textMuted, fontSize: 9, letterSpacing: 1 },
  gridButton: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.surface, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: Colors.border },
  
  listContent: { padding: 15 },
  cameraCard: { backgroundColor: Colors.surface, borderRadius: 16, marginBottom: 20, overflow: 'hidden', borderWidth: 1, borderColor: Colors.border },
  cameraHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 12, backgroundColor: 'rgba(255,255,255,0.03)' },
  cameraName: { color: Colors.text, fontSize: 12, fontWeight: '700', letterSpacing: 0.5 },
  badgeContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.success + '20', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4 },
  dot: { width: 4, height: 4, borderRadius: 2, marginRight: 4 },
  badgeText: { color: Colors.success, fontSize: 8, fontWeight: '900' },
  
  streamContainer: { width: '100%', height: 220, backgroundColor: '#000', position: 'relative' },
  stream: { width: '100%', height: '100%', opacity: 0.8 },
  expandButton: { position: 'absolute', top: 10, right: 10, backgroundColor: 'rgba(0,0,0,0.5)', padding: 8, borderRadius: 8 },
  hudOverlay: { position: 'absolute', bottom: 10, left: 10, backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  hudText: { color: 'rgba(255,255,255,0.7)', fontSize: 8, fontWeight: 'bold', fontFamily: 'System' },
  
  alertOverlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center' },
  boundingBox: { width: 100, height: 150, borderWidth: 2, borderColor: Colors.destructive, borderRadius: 4 },
  alertTag: { position: 'absolute', top: 30, backgroundColor: Colors.destructive, flexDirection: 'row', alignItems: 'center', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  alertTagText: { color: '#fff', fontSize: 10, fontWeight: 'bold', marginLeft: 4 },

  cameraFooter: { flexDirection: 'row', padding: 15, borderTopWidth: 1, borderTopColor: Colors.border },
  footerStat: { flex: 1, alignItems: 'center' },
  statLabel: { color: Colors.textMuted, fontSize: 8, fontWeight: '700', marginBottom: 2 },
  statValue: { color: Colors.text, fontSize: 14, fontWeight: 'bold' },
});
