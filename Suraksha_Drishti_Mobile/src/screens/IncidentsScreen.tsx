import React from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ShieldAlert, Clock, MapPin, CheckCircle2, ChevronRight, Share2 } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { useAppStore } from '../store/useAppStore';
import { LinearGradient } from 'expo-linear-gradient';

export function IncidentsScreen() {
  const { incidents } = useAppStore();

  const renderItem = ({ item }: { item: any }) => (
    <TouchableOpacity style={styles.card}>
      <View style={[styles.priorityIndicator, { backgroundColor: item.level === 'CRITICAL' ? Colors.destructive : Colors.warning }]} />
      
      <View style={styles.cardContent}>
        <View style={styles.cardHeader}>
          <View style={styles.idContainer}>
            <Text style={styles.incidentId}>{item.id}</Text>
            <View style={[styles.levelBadge, { backgroundColor: item.level === 'CRITICAL' ? Colors.destructive + '20' : Colors.warning + '20' }]}>
              <Text style={[styles.levelText, { color: item.level === 'CRITICAL' ? Colors.destructive : Colors.warning }]}>{item.level}</Text>
            </View>
          </View>
          <Text style={styles.timeText}>{new Date(item.ts).toLocaleTimeString()}</Text>
        </View>

        <Text style={styles.summaryText}>{item.summary}</Text>

        <View style={styles.metaRow}>
          <View style={styles.metaItem}>
            <MapPin size={12} color={Colors.textMuted} />
            <Text style={styles.metaText}>{item.location}</Text>
          </View>
          <View style={styles.metaItem}>
            <Clock size={12} color={Colors.textMuted} />
            <Text style={styles.metaText}>{new Date(item.ts).toLocaleDateString()}</Text>
          </View>
        </View>

        <View style={styles.actions}>
          <TouchableOpacity style={styles.actionBtn}>
            <Share2 size={16} color={Colors.primary} />
            <Text style={styles.actionBtnText}>DISPATCH</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.actionBtn, styles.resolveBtn]}>
            <CheckCircle2 size={16} color={Colors.success} />
            <Text style={[styles.actionBtnText, { color: Colors.success }]}>RESOLVE</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.detailsBtn}>
            <ChevronRight size={20} color={Colors.textMuted} />
          </TouchableOpacity>
        </View>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={[Colors.surface, Colors.background]} style={StyleSheet.absoluteFill} />
      
      <View style={styles.header}>
        <Text style={styles.headerTitle}>INCIDENT LOGS</Text>
        <Text style={styles.headerSubtitle}>FORENSIC EVIDENCE & RESPONSE</Text>
      </View>

      {incidents.length > 0 ? (
        <FlatList
          data={incidents}
          renderItem={renderItem}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.listContent}
        />
      ) : (
        <View style={styles.emptyContainer}>
          <ShieldAlert size={64} color={Colors.surfaceLight} />
          <Text style={styles.emptyTitle}>NO ACTIVE THREATS</Text>
          <Text style={styles.emptySubtitle}>The sector is currently secure.</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { padding: 20, borderBottomWidth: 1, borderBottomColor: Colors.border },
  headerTitle: { color: Colors.primary, fontSize: 18, fontWeight: '900', letterSpacing: 2 },
  headerSubtitle: { color: Colors.textMuted, fontSize: 10, letterSpacing: 1, marginTop: 4 },
  
  listContent: { padding: 15 },
  card: { 
    flexDirection: 'row', 
    backgroundColor: Colors.surface, 
    borderRadius: 16, 
    marginBottom: 15, 
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8
  },
  priorityIndicator: { width: 5 },
  cardContent: { flex: 1, padding: 15 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 },
  idContainer: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  incidentId: { color: Colors.primary, fontSize: 12, fontWeight: 'bold', fontFamily: 'System' },
  levelBadge: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4 },
  levelText: { fontSize: 8, fontWeight: '900' },
  timeText: { color: Colors.textMuted, fontSize: 10, fontWeight: '600' },
  
  summaryText: { color: Colors.text, fontSize: 14, fontWeight: '600', lineHeight: 20, marginBottom: 12 },
  
  metaRow: { flexDirection: 'row', gap: 20, marginBottom: 15 },
  metaItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  metaText: { color: Colors.textMuted, fontSize: 10, fontWeight: '500' },
  
  actions: { flexDirection: 'row', alignItems: 'center', gap: 10, borderTopWidth: 1, borderTopColor: Colors.border, paddingTop: 12 },
  actionBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: Colors.primary + '10', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: Colors.primary + '30' },
  resolveBtn: { backgroundColor: Colors.success + '10', borderColor: Colors.success + '30' },
  actionBtnText: { color: Colors.primary, fontSize: 10, fontWeight: '900', letterSpacing: 0.5 },
  detailsBtn: { marginLeft: 'auto' },

  emptyContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingBottom: 100 },
  emptyTitle: { color: Colors.text, fontSize: 16, fontWeight: '900', letterSpacing: 2, marginTop: 20 },
  emptySubtitle: { color: Colors.textMuted, fontSize: 12, marginTop: 8 },
});
