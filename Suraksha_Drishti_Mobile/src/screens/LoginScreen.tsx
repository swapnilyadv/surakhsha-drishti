import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, Image, KeyboardAvoidingView, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Shield, Fingerprint, Lock, User, ChevronRight } from 'lucide-react-native';
import { Colors } from '../theme/theme';
import { LinearGradient } from 'expo-linear-gradient';
import { supabase } from '../api/supabase';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

export function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={['#020617', '#0f172a', '#020617']} style={StyleSheet.absoluteFill} />
      
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.inner}
      >
        <Animated.View entering={FadeIn.delay(300)} style={styles.logoContainer}>
          <View style={styles.logoHex}>
            <Shield size={50} color={Colors.primary} />
          </View>
          <Text style={styles.logoText}>SURAKSHA DRISHTI</Text>
          <Text style={styles.logoTagline}>AI TACTICAL COMMAND</Text>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(600)} style={styles.form}>
          <View style={styles.inputContainer}>
            <User size={18} color={Colors.primary} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="OFFICER ID"
              placeholderTextColor="rgba(56, 189, 248, 0.4)"
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
            />
          </View>

          <View style={styles.inputContainer}>
            <Lock size={18} color={Colors.primary} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="ACCESS CODE"
              placeholderTextColor="rgba(56, 189, 248, 0.4)"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>

          <TouchableOpacity
            style={styles.loginBtn}
            disabled={loading}
            onPress={async () => {
              setLoading(true);
              setError(null);
              try {
                // Supabase expects an email by default; adapt as needed.
                const identifier = username.trim();
                const { data, error: signInError } = await supabase.auth.signInWithPassword({
                  email: identifier,
                  password: password,
                });

                if (signInError) {
                  setError(signInError.message || 'Failed to sign in');
                } else {
                  // Signed in successfully. You can navigate or store session as needed.
                  // For now, just log the user.
                  // eslint-disable-next-line no-console
                  console.log('Signed in:', data);
                }
              } catch (err: any) {
                setError(err?.message || String(err));
              } finally {
                setLoading(false);
              }
            }}
          >
            <LinearGradient
              colors={[Colors.primary, Colors.accent]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.btnGradient}
            >
              <Text style={styles.loginBtnText}>INITIALIZE SESSION</Text>
              <ChevronRight size={20} color="#fff" />
            </LinearGradient>
          </TouchableOpacity>

          {error ? <Text style={{ color: 'red', marginTop: 8 }}>{error}</Text> : null}

          <TouchableOpacity style={styles.biometricBtn}>
            <Fingerprint size={32} color={Colors.primary} />
            <Text style={styles.biometricText}>BIOMETRIC ACCESS</Text>
          </TouchableOpacity>
        </Animated.View>

        <Animated.View entering={FadeIn.delay(1000)} style={styles.footer}>
          <Text style={styles.footerText}>SECURE TERMINAL CONNECTION</Text>
          <View style={styles.line} />
          <Text style={styles.encryptionText}>AES-256 ENCRYPTION ACTIVE</Text>
        </Animated.View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#020617' },
  inner: { flex: 1, padding: 30, justifyContent: 'center' },
  
  logoContainer: { alignItems: 'center', marginBottom: 60 },
  logoHex: { 
    width: 100, 
    height: 100, 
    backgroundColor: Colors.surface, 
    justifyContent: 'center', 
    alignItems: 'center', 
    borderRadius: 20,
    borderWidth: 2,
    borderColor: Colors.primary,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    marginBottom: 20
  },
  logoText: { color: Colors.text, fontSize: 24, fontWeight: '900', letterSpacing: 4 },
  logoTagline: { color: Colors.primary, fontSize: 10, fontWeight: 'bold', letterSpacing: 2, marginTop: 8 },
  
  form: { gap: 15 },
  inputContainer: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    backgroundColor: Colors.surface, 
    borderRadius: 12, 
    borderWidth: 1, 
    borderColor: Colors.border,
    paddingHorizontal: 15
  },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, height: 55, color: Colors.text, fontSize: 14, fontWeight: '600', letterSpacing: 1 },
  
  loginBtn: { height: 55, borderRadius: 12, overflow: 'hidden', marginTop: 10 },
  btnGradient: { flex: 1, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 10 },
  loginBtnText: { color: '#fff', fontSize: 14, fontWeight: '900', letterSpacing: 2 },
  
  biometricBtn: { alignItems: 'center', marginTop: 30, gap: 10 },
  biometricText: { color: Colors.primary, fontSize: 10, fontWeight: 'bold', letterSpacing: 1 },
  
  footer: { position: 'absolute', bottom: 50, left: 0, right: 0, alignItems: 'center' },
  footerText: { color: Colors.textMuted, fontSize: 8, fontWeight: 'bold', letterSpacing: 2 },
  line: { width: 100, height: 1, backgroundColor: Colors.border, marginVertical: 10 },
  encryptionText: { color: Colors.success, fontSize: 8, fontWeight: 'bold', letterSpacing: 1 },
});
