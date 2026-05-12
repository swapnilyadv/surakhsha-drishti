import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Shield, ShieldAlert, Activity, Users, Zap } from 'lucide-react-native';
import { Colors, Typography } from '../theme/theme';
import { useAppStore } from '../store/useAppStore';
import Animated, { FadeInUp, FadeInRight } from 'react-native-reanimated';

const { width } = Dimensions.get('window');

export function DashboardScreen() {
  const { isConnected, threatLevel, incidents, detections } = useAppStore();

  const totalPeople = Object.values(detections).reduce((sum, d) => sum + (d.crowd_count || 0), 0);
  const activeAlerts = Object.values(detections).filter(d => d.violence || d.weapon).length;

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={[Colors.surface, Colors.background]}
        style={StyleSheet.absoluteFill}
      />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <Animated.View entering={FadeInUp.delay(200)} style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>COMMAND CENTER</Text>
            <Text style={styles.headerSubtitle}>SURAKSHA DRISHTI MOBILE</Text>
          </View>
          <View style={[styles.statusBadge, isConnected ? styles.statusOnline : styles.statusOffline]}>
            <View style={[styles.statusDot, isConnected ? styles.dotOnline : styles.dotOffline]} />
            <Text style={styles.statusText}>{isConnected ? 'LIVE' : 'OFFLINE'}</Text>
          </View>
        </Animated.View>

        {/* Threat Level Banner */}
        <Animated.View entering={FadeInUp.delay(400)} style={[styles.threatBanner, styles[`threat${threatLevel}`]]}>
          <ShieldAlert color="#fff" size={32} />
          <View style={styles.threatTextContainer}>
            <Text style={styles.threatLabel}>CURRENT THREAT LEVEL</Text>
            <Text style={styles.threatValue}>{threatLevel}</Text>
          </View>
          <Zap color="#fff" size={24} style={styles.zapIcon} />
        </Animated.View>

        {/* Stats Grid */}
        <View style={styles.statsGrid}>
          <StatCard 
            label="ACTIVE ALERTS" 
            value={activeAlerts} 
            icon={<ShieldAlert size={20} color={Colors.destructive} />} 
            delay={600}
            color={Colors.destructive}
          />
          <StatCard 
            label="CROWD SIZE" 
            value={totalPeople} 
            icon={<Users size={20} color={Colors.primary} />} 
            delay={700}
            color={Colors.primary}
          />
          <StatCard 
            label="INCIDENTS" 
            value={incidents.length} 
            icon={<Activity size={20} color={Colors.warning} />} 
            delay={800}
            color={Colors.warning}
          />
          <StatCard 
            label="NODES" 
            value={Object.keys(detections).length || 0} 
            icon={<Shield size={20} color={Colors.success} />} 
            delay={900}
            color={Colors.success}
          />
        </View>

        {/* Recent Activity */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>RECENT ACTIVITY</Text>
          <TouchableOpacity>
            <Text style={styles.viewAll}>VIEW ALL</Text>
          </TouchableOpacity>
        </View>

        {incidents.length > 0 ? (
          incidents.slice(0, 3).map((inc, i) => (
            <Animated.View key={inc.id} entering={FadeInRight.delay(1000 + (i * 100))} style={styles.incidentCard}>
              <View style={[styles.incidentLevel, { backgroundColor: inc.level === 'CRITICAL' ? Colors.destructive : Colors.warning }]} />
              <View style={styles.incidentInfo}>
                <Text style={styles.incidentTitle}>{inc.summary}</Text>
                <Text style={styles.incidentMeta}>{inc.location} • {new Date(inc.ts).toLocaleTimeString()}</Text>
              </View>
            </Animated.View>
          ))
        ) : (
          <Text style={styles.emptyText}>No recent threats detected.</Text>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function StatCard({ label, value, icon, delay, color }: any) {
  return (
    <Animated.View entering={FadeInUp.delay(delay)} style={styles.statCard}>
      <View style={[styles.statIconContainer, { backgroundColor: color + '20' }]}>
        {icon}
      </View>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  scrollContent: { padding: 20 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25 },
  headerTitle: { color: Colors.primary, fontSize: 18, fontWeight: '900', letterSpacing: 2 },
  headerSubtitle: { color: Colors.textMuted, fontSize: 10, letterSpacing: 1 },
  statusBadge: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12, borderWidth: 1 },
  statusOnline: { borderColor: Colors.success + '40', backgroundColor: Colors.success + '10' },
  statusOffline: { borderColor: Colors.textMuted + '40', backgroundColor: Colors.textMuted + '10' },
  statusDot: { width: 6, height: 6, borderRadius: 3, marginRight: 6 },
  dotOnline: { backgroundColor: Colors.success },
  dotOffline: { backgroundColor: Colors.textMuted },
  statusText: { color: Colors.text, fontSize: 10, fontWeight: '700' },
  
  threatBanner: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    padding: 20, 
    borderRadius: 16, 
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5
  },
  threatSAFE: { backgroundColor: Colors.success },
  threatLOW: { backgroundColor: Colors.accent },
  threatHIGH: { backgroundColor: Colors.warning },
  threatCRITICAL: { backgroundColor: Colors.destructive },
  threatTextContainer: { marginLeft: 15, flex: 1 },
  threatLabel: { color: 'rgba(255,255,255,0.7)', fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  threatValue: { color: '#fff', fontSize: 24, fontWeight: '900', letterSpacing: 2 },
  zapIcon: { opacity: 0.5 },

  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 25 },
  statCard: { 
    width: (width - 55) / 2, 
    backgroundColor: Colors.surface, 
    padding: 15, 
    borderRadius: 16, 
    marginBottom: 15,
    borderWidth: 1,
    borderColor: Colors.border
  },
  statIconContainer: { width: 36, height: 36, borderRadius: 10, justifyContent: 'center', alignItems: 'center', marginBottom: 10 },
  statValue: { color: Colors.text, fontSize: 20, fontWeight: 'bold' },
  statLabel: { color: Colors.textMuted, fontSize: 10, fontWeight: '600', letterSpacing: 0.5 },

  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  sectionTitle: { color: Colors.primary, fontSize: 12, fontWeight: '800', letterSpacing: 1.5 },
  viewAll: { color: Colors.textMuted, fontSize: 10, fontWeight: '600' },
  
  incidentCard: { 
    flexDirection: 'row', 
    backgroundColor: Colors.surface, 
    borderRadius: 12, 
    marginBottom: 10, 
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border
  },
  incidentLevel: { width: 4 },
  incidentInfo: { padding: 15, flex: 1 },
  incidentTitle: { color: Colors.text, fontSize: 13, fontWeight: '600', marginBottom: 4 },
  incidentMeta: { color: Colors.textMuted, fontSize: 10 },
  emptyText: { color: Colors.textMuted, fontSize: 12, textAlign: 'center', marginTop: 20, fontStyle: 'italic' },
});
