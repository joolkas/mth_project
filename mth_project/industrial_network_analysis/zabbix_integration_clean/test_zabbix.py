#!/usr/bin/env python3
"""
🧪 Simple Zabbix Connection Test

Quick script to test basic Zabbix connectivity without loading ML models.
"""

import json
import sys
from pyzabbix import ZabbixAPI
import urllib3

# Disable SSL warnings
urllib3.disable_warnings()

def test_zabbix_connection(config_file="config.json"):
    """Test basic Zabbix connection"""
    
    print("🔍 Loading configuration...")
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        zabbix_config = config['zabbix']
        print(f"   URL: {zabbix_config['url']}")
        print(f"   User: {zabbix_config['user']}")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    print("\n🔗 Testing Zabbix connection...")
    try:
        # Create API connection
        zapi = ZabbixAPI(zabbix_config['url'])
        zapi.session.verify = False
        
        print("🔐 Attempting login...")
        zapi.login(zabbix_config['user'], zabbix_config['password'])
        
        # Get version info
        version = zapi.apiinfo.version()
        print(f"✅ Connected to Zabbix {version}")
        
        # Test getting host groups
        print("\n📋 Testing host group access...")
        groups = zapi.hostgroup.get(output=['groupid', 'name'])
        print(f"   Found {len(groups)} host groups:")
        for group in groups[:10]:  # Show first 10
            print(f"   - {group['name']} (ID: {group['groupid']})")
        
        # Test configured groups
        configured_groups = config['monitoring']['host_groups']
        print(f"\n🎯 Checking configured groups: {configured_groups}")
        
        found_groups = []
        for group_name in configured_groups:
            matching = [g for g in groups if group_name.lower() in g['name'].lower()]
            if matching:
                found_groups.extend(matching)
                print(f"   ✅ Found: {matching[0]['name']}")
            else:
                print(f"   ⚠️ Not found: {group_name}")
        
        if found_groups:
            # Test getting hosts from found groups
            print(f"\n🖥️ Testing host access...")
            for group in found_groups[:3]:  # Test first 3 groups
                hosts = zapi.host.get(
                    groupids=[group['groupid']],
                    output=['hostid', 'host', 'name']
                )
                print(f"   Group '{group['name']}': {len(hosts)} hosts")
                for host in hosts[:3]:  # Show first 3 hosts
                    print(f"     - {host['host']} ({host['name']})")
        
        print("\n✅ Zabbix connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Zabbix connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if Zabbix server is running")
        print("   2. Verify URL format: http://<server>/zabbix")
        print("   3. Test URL in browser")
        print("   4. Check username/password")
        print("   5. Ensure user has API access permissions")
        return False

if __name__ == "__main__":
    print("🧪 Zabbix Connection Test")
    print("=" * 40)
    
    config_file = "config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    success = test_zabbix_connection(config_file)
    sys.exit(0 if success else 1)