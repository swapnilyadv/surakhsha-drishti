import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Switch, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Settings, Shield, Bell, Siren, LogOut, ChevronRight, User } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { LinearGradient } from 'expo-linear-gradient';

export function SettingsScreen() {
  const [sirenEnabled, setSirenEnabled] = useState(true);
  const [notifications, setNotifications] = useState(true);

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={[Colors.surface, Colors.background]} style={StyleSheet.absoluteFill} />
      
      <View style={styles.header}>
        <Text style={styles.headerTitle}>SYSTEM SETTINGS</Text>
        <Text style={styles.headerSubtitle}>ADMINISTRATIVE CONTROLS</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Profile */}
        <TouchableOpacity style={styles.profileCard}>
          <View style={styles.avatar}>
            <User size={30} color={Colors.primary} />
          </View>
          <View style={styles.profileInfo}>
            <Text style={styles.profileName}>Admin Officer</Text>
            <Text style={styles.profileRole}>Tactical Response Team</Text>
          </View>
          <ChevronRight size={20} color={Colors.textMuted} />
        </TouchableOpacity>

        <Section title="AI MONITORING">
          <SettingRow 
            icon={<Shield size={20} color={Colors.primary} />} 
            label="Violence Sensitivity" 
            value="85%" 
          />
          <SettingRow 
            icon={<Shield size={20} color={Colors.primary} />} 
            label="Weapon Detection" 
            value="ACTIVE" 
          />
        </Section>

        <Section title="ALERTS & NOTIFICATIONS">
          <SettingToggle 
            icon={<Bell size={20} color={Colors.warning} />} 
            label="Push Notifications" 
            value={notifications}
            onToggle={setNotifications}
          />
          <SettingToggle 
            icon={<Siren size={20} color={Colors.destructive} />} 
            label="Emergency Siren" 
            value={sirenEnabled}
            onToggle={setSirenEnabled}
          />
        </Section>

        <Section title="SECURITY">
          <SettingRow 
            icon={<Shield size={20} color={Colors.success} />} 
            label="Biometric FaceID" 
            value="ENABLED" 
          />
          <SettingRow 
            icon={<Settings size={20} color={Colors.textMuted} />} 
            label="System Version" 
            value="v2.4.0-tactical" 
          />
        </Section>

        <TouchableOpacity style={styles.logoutBtn}>
          <LogOut size={20} color={Colors.destructive} />
          <Text style={styles.logoutText}>TERMINATE SESSION</Text>
        </TouchableOpacity>
        
        <Text style={styles.footer}>© 2026 SURAKSHA DRISHTI AI CORE</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children }: any) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <View style={styles.sectionContent}>{children}</View>
    </View>
  );
}

function SettingRow({ icon, label, value }: any) {
  return (
    <TouchableOpacity style={styles.row}>
      <View style={styles.rowLeft}>
        {icon}
        <Text style={styles.rowLabel}>{label}</Text>
      </View>
      <Text style={styles.rowValue}>{value}</Text>
    </TouchableOpacity>
  );
}

function SettingToggle({ icon, label, value, onToggle }: any) {
  return (
    <View style={styles.row}>
      <View style={styles.rowLeft}>
        {icon}
        <Text style={styles.rowLabel}>{label}</Text>
      </View>
      <Switch 
        value={value} 
        onValueChange={onToggle}
        trackColor={{ false: Colors.surfaceLight, true: Colors.primary }}
        thumbColor="#fff"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { padding: 20, borderBottomWidth: 1, borderBottomColor: Colors.border },
  headerTitle: { color: Colors.primary, fontSize: 18, fontWeight: '900', letterSpacing: 2 },
  headerSubtitle: { color: Colors.textMuted, fontSize: 10, letterSpacing: 1, marginTop: 4 },
  
  content: { padding: 20 },
  profileCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.surface, padding: 20, borderRadius: 16, marginBottom: 25, borderWidth: 1, borderColor: Colors.border },
  avatar: { width: 50, height: 50, borderRadius: 25, backgroundColor: Colors.primary + '20', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: Colors.primary + '40' },
  profileInfo: { flex: 1, marginLeft: 15 },
  profileName: { color: Colors.text, fontSize: 16, fontWeight: 'bold' },
  profileRole: { color: Colors.textMuted, fontSize: 10, fontWeight: '600', marginTop: 2 },
  
  section: { marginBottom: 25 },
  sectionTitle: { color: Colors.primary, fontSize: 10, fontWeight: '900', letterSpacing: 1.5, marginBottom: 12, marginLeft: 5 },
  sectionContent: { backgroundColor: Colors.surface, borderRadius: 16, overflow: 'hidden', borderWidth: 1, borderColor: Colors.border },
  
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16, borderBottomWidth: 1, borderBottomColor: Colors.border },
  rowLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  rowLabel: { color: Colors.text, fontSize: 13, fontWeight: '600' },
  rowValue: { color: Colors.primary, fontSize: 12, fontWeight: 'bold' },
  
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, backgroundColor: Colors.destructive + '10', padding: 18, borderRadius: 16, marginTop: 10, borderWidth: 1, borderColor: Colors.destructive + '30' },
  logoutText: { color: Colors.destructive, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  
  footer: { textAlign: 'center', color: Colors.textMuted, fontSize: 9, fontWeight: 'bold', marginTop: 30, letterSpacing: 1 },
});
