#!/usr/bin/env python3
"""
Unity 6 Sentis UPM Packager for EdgeStereoINR.
Generates compliant Unity .meta files and packages the directory into
a standard .tgz tarball for Unity Package Manager (UPM) distribution.
"""

import os
import hashlib
import tarfile
import json

UPM_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(UPM_DIR, "dist")

def generate_guid(filepath: str) -> str:
    """Generates a deterministic 32-character hex GUID from the relative path."""
    rel = os.path.relpath(filepath, UPM_DIR).replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()

def create_meta_file(filepath: str):
    meta_path = filepath + ".meta"
    if os.path.exists(meta_path):
        return
    
    guid = generate_guid(filepath)
    is_dir = os.path.isdir(filepath)
    
    if is_dir:
        content = f"""fileFormatVersion: 2
guid: {guid}
folderAsset: yes
DefaultImporter:
  externalObjects: {{}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""
    elif filepath.endswith(".cs"):
        content = f"""fileFormatVersion: 2
guid: {guid}
MonoImporter:
  externalObjects: {{}}
  serializedVersion: 2
  defaultReferences: []
  executionOrder: 0
  icon: {{instanceID: 0}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""
    elif filepath.endswith(".shader"):
        content = f"""fileFormatVersion: 2
guid: {guid}
ShaderImporter:
  externalObjects: {{}}
  defaultTextures: []
  nonModifiableTextures: []
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""
    else:
        content = f"""fileFormatVersion: 2
guid: {guid}
DefaultImporter:
  externalObjects: {{}}
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[*] Generated: {os.path.basename(meta_path)} (GUID: {guid})")

def build_upm_package():
    os.makedirs(DIST_DIR, exist_ok=True)
    
    # 1. Ensure all files have .meta files
    for root, dirs, files in os.walk(UPM_DIR):
        if "dist" in root or ".git" in root or "__pycache__" in root:
            continue
        for d in dirs:
            if d != "dist":
                create_meta_file(os.path.join(root, d))
        for f in files:
            if not f.endswith(".meta") and not f.endswith(".tgz"):
                create_meta_file(os.path.join(root, f))
                
    # 2. Read package.json for metadata
    pkg_json_path = os.path.join(UPM_DIR, "package.json")
    with open(pkg_json_path, "r", encoding="utf-8") as f:
        pkg_data = json.load(f)
        
    pkg_name = pkg_data.get("name", "com.activestereo.edgestreoinr")
    pkg_version = pkg_data.get("version", "1.0.0")
    tarball_name = f"{pkg_name}-{pkg_version}.tgz"
    tarball_path = os.path.join(DIST_DIR, tarball_name)
    
    # 3. Create tarball with root directory named "package" (npm/UPM standard)
    with tarfile.open(tarball_path, "w:gz") as tar:
        for root, dirs, files in os.walk(UPM_DIR):
            if "dist" in root or ".git" in root or "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py") or f.endswith(".tgz"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, UPM_DIR).replace("\\", "/")
                arcname = f"package/{rel_path}"
                tar.add(full_path, arcname=arcname)
                
    tar_size_kb = os.path.getsize(tarball_path) / 1024.0
    print("=" * 70)
    print(f"[+] Successfully built Unity UPM Tarball: {tarball_path}")
    print(f"[+] Tarball Size: {tar_size_kb:.2f} KB")
    print(f"[+] UPM Install string: file:{os.path.abspath(tarball_path).replace(chr(92), '/')}")
    print("=" * 70)

if __name__ == "__main__":
    build_upm_package()
