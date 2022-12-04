import React from "react";

const Voxel = () => {
  const width = 100;
  const height = 100;
  const depth = 100;

  const voxelData = [];
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      for (let z = 0; z < depth; z++) {
        const value = Math.random();
        voxelData.push({ x, y, z, value });
      }
    }
  }

  const voxelGeometry = new THREE.BoxGeometry(1, 1, 1);
  const voxelMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });

  const voxels = [];
  for (const voxel of voxelData) {
    const voxelMesh = new THREE.Mesh(voxelGeometry, voxelMaterial);
    voxelMesh.position.set(voxel.x, voxel.y, voxel.z);
    voxels.push(voxelMesh);
  }

  const scene = new THREE.Scene();
  for (const voxel of voxels) {
    scene.add(voxel);
  }

  const renderer = new THREE.WebGLRenderer();
  document.body.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 500;

  const animate = () => {
    requestAnimationFrame(animate);

    renderer.render(scene, camera);
  };

  animate();

  return <div></div>;
};

export default Voxel;
