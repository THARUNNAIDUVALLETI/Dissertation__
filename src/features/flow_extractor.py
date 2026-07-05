"""Extract network flow features from packets"""
from collections import defaultdict
import time
import numpy as np

class FlowExtractor:
    def __init__(self, timeout=60):
        self.flows = defaultdict(list)
        self.timeout = timeout
        
    def extract_flow_key(self, packet):
        if not hasattr(packet, 'ip'): 
            return None
        proto = 'tcp' if hasattr(packet, 'tcp') else 'udp' if hasattr(packet, 'udp') else 'icmp'
        sport = getattr(packet, proto).sport if proto in ['tcp','udp'] else 0
        dport = getattr(packet, proto).dport if proto in ['tcp','udp'] else 0
        return (packet.ip.src, packet.ip.dst, sport, dport, proto)
    
    def extract_features(self, flow_packets):
        if not flow_packets: 
            return None
        durations = []
        sizes = []
        flags = {'syn':0, 'ack':0, 'fin':0, 'rst':0}
        for i, pkt in enumerate(flow_packets):
            if i > 0:
                dur = pkt.time - flow_packets[i-1].time
                durations.append(dur)
            sizes.append(len(pkt))
            if hasattr(pkt, 'tcp'):
                if pkt.tcp.flags.syn: flags['syn'] += 1
                if pkt.tcp.flags.ack: flags['ack'] += 1
                if pkt.tcp.flags.fin: flags['fin'] += 1
                if pkt.tcp.flags.rst: flags['rst'] += 1
        features = {
            'duration': flow_packets[-1].time - flow_packets[0].time,
            'packet_count': len(flow_packets),
            'byte_total': sum(sizes),
            'byte_mean': np.mean(sizes) if sizes else 0,
            'byte_std': np.std(sizes) if sizes else 0,
            'iat_mean': np.mean(durations) if durations else 0,
            'iat_std': np.std(durations) if durations else 0,
            'syn_count': flags['syn'],
            'ack_count': flags['ack'],
            'fin_count': flags['fin'],
            'rst_count': flags['rst'],
        }
        return features
    
    def add_packet(self, packet):
        key = self.extract_flow_key(packet)
        if key:
            self.flows[key].append(packet)
        self._cleanup_old_flows()
    
    def _cleanup_old_flows(self):
        current_time = time.time()
        expired = [key for key, packets in self.flows.items() 
                  if current_time - packets[-1].time > self.timeout]
        for key in expired:
            del self.flows[key]
