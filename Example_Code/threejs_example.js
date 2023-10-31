// From: https://codepen.io/dev_loop/pen/LYNpPWj

const vertexShader = `
		varying vec2 vUv;

		uniform float twistAmount;

		mat4 rotation3d(vec3 axis, float angle) {
		axis = normalize(axis);
		float s = sin(angle);
		float c = cos(angle);
		float oc = 1.0 - c;

		return mat4(
		oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
		oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
		oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
		0.0,                                0.0,                                0.0,                                1.0
		);
		}

		vec3 rotate(vec3 v, vec3 axis, float angle) {
		return (rotation3d(axis, angle) * vec4(v, 1.0)).xyz;
		}

		void main() {
		vUv = uv;

		vec3 pos = position;

		vec3 axis = vec3(1., 0., 0.);
		float twist = -twistAmount;
		float angle = pos.x * twist;

		vec3 transformed = rotate(pos, axis, angle);

		gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.);
		}
`;

const fragMentShader =  `
		varying vec2 vUv;

		uniform sampler2D uTexture;

		void main() {
		vec2 uv = fract(vUv * 1. - vec2(0.05, 0.));
		vec3 texture = texture2D(uTexture, uv).rgb;

		gl_FragColor = vec4(texture, 1.);
		}
`;


// ******** SCENE CODE ************* //

class GL {
	constructor() {
		this.width = window.innerWidth;
		this.height = window.innerHeight;

		this.createScene();
		this.createCamera();

		this.init();
	}

	createScene() {
		this.renderer = new THREE.WebGLRenderer({
			alpha: true,
			antialias: true,
		});
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
		this.renderer.setSize(this.width, this.height);
		this.renderer.setClearColor(0x001845, 1);

		this.scene = new THREE.Scene();
	}

	createCamera() {
		this.camera = new THREE.PerspectiveCamera(
			45,
			this.width / this.height,
			1,
			1000
		);

		this.camera.position.z = 5;

		this.camera.lookAt(new THREE.Vector3());
	}

	render() {
		this.renderer.render(this.scene, this.camera);
	}

	animate() {
		requestAnimationFrame(this.animate.bind(this));

		this.render();
	}

	addEvents() {
		window.addEventListener("resize", this.resize.bind(this));
	}

	init() {
		this.addToDom();
		this.animate();
		this.addEvents();
	}

	addToDom() {
		const canvas = this.renderer.domElement;
		document.body.appendChild(canvas);
	}

	resize() {
		this.width = window.innerWidth;
		this.height = window.innerHeight;

		this.camera.aspect = this.width / this.height;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(this.width, this.height);
	}
}

// INIT THE SCENE 
const _GL = new GL();


// ******** THE TYPE CODE (ohh my God GSAP is sooo fun) ************* //

class Type extends THREE.Object3D {
	init(options) {
		this.opts = {
			word: options.word,
			color: options.color,
			fill: options.fill,
			wordPosition: options.position.texture,
			wordScale: options.scale,
			position: options.position.mesh,
			rotation: options.rotation || [0, 0, 0],
			geometry: options.geometry,
			vertex: options.shaders.vertex,
			fragment: options.shaders.fragment,
			fontFile: options.font.file,
			fontAtlas: options.font.atlas,
		};

		// Create geometry of packed glyphs
		loadBmfont(options.font.file, (err, font) => {
			this.fontGeometry = threeBmfontText({
				font,
				text: this.opts.word,
			});

			// Load texture containing font glyps
			this.loader = new THREE.TextureLoader();
			this.loader.crossOrigin = 'Anonymous';
			this.loader.load(this.opts.fontAtlas, (texture) => {
				this.fontMaterial = new THREE.RawShaderMaterial(
					createMSDFShader ({
						map: texture,
						side: THREE.DoubleSide,
						transparent: true,
						negate: false,
						color: this.opts.color,
					})
				);

				this.createRenderTarget();
				this.createMesh();
				this.twistAndFlip();
			});
		});
	}

	createRenderTarget() {
		this.rt = new THREE.WebGLRenderTarget(
			window.innerWidth,
			window.innerHeight
		);
		this.rtCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
		this.rtCamera.position.z = 2.3;

		this.rtScene = new THREE.Scene();
		this.rtScene.background = new THREE.Color(this.opts.fill);

		this.text = new THREE.Mesh(this.fontGeometry, this.fontMaterial);
		this.text.position.set(...this.opts.wordPosition);
		this.text.rotation.set(Math.PI, 0, 0);
		this.text.scale.set(...this.opts.wordScale);

		this.rtScene.add(this.text);
	}

	createMesh() {
		this.geometry = this.opts.geometry;

		this.material = new THREE.ShaderMaterial({
			vertexShader: this.opts.vertex,
			fragmentShader: this.opts.fragment,
			uniforms: {
				twistAmount: { value: 0 },
				uTexture: { value: this.rt.texture },
			},
			defines: {
				PI: Math.PI,
			},
			side: THREE.DoubleSide,
		});

		this.mesh = new THREE.Mesh(this.geometry, this.material);
		this.mesh.position.set(...this.opts.position);
		this.mesh.rotation.set(...this.opts.rotation);
		this.mesh.lookAt(new THREE.Vector3());

		this.mesh.onBeforeRender = (renderer) => {
			renderer.setRenderTarget(this.rt);
			renderer.render(this.rtScene, this.rtCamera);
			renderer.setRenderTarget(null);
		};

		this.add(this.mesh);

		_GL.scene.add(this);
	}

	twistAndFlip() {
		const twistAmount = 1;
		const rotationAngleFactor = 58;

		const tlDefaults = {
			duration: 0.5,
			ease: "elastic.out(0.8, 1.5)",
		};

		const tl = gsap.timeline({
			delay: 1,
			repeat: -1,
			repeatDelay: 0.5,
			defaults: tlDefaults,
		});

		tl.addLabel("start", 0);
		tl.addLabel("startEnd", tlDefaults.duration);
		tl.addLabel("endStart", tlDefaults.duration * 2 + 0.5);
		tl.addLabel("end", tlDefaults.duration * 3 + 0.5);

		tl.to(
			this.material.uniforms.twistAmount,
			{
				value: twistAmount,
			},
			"start"
		)
			.to(
			this.mesh.rotation,
			{
				x: THREE.MathUtils.degToRad(
					twistAmount * rotationAngleFactor
				),
			},
			"start"
		)
			.to(
			this.material.uniforms.twistAmount,
			{
				value: 0,
			},
			"startEnd"
		)
			.to(
			this.mesh.rotation,
			{
				x: this.mesh.rotation.x - THREE.MathUtils.degToRad(90),
			},
			"startEnd"
		)
			.to(
			this.material.uniforms.twistAmount,
			{
				value: twistAmount,
			},
			"endStart"
		)
			.to(
			this.mesh.rotation,
			{
				x:
				this.mesh.rotation.x -
				THREE.MathUtils.degToRad(
					twistAmount * rotationAngleFactor * 0.55
				),
			},
			"endStart"
		)
			.set(
			this.mesh.rotation,
			{
				z: Math.PI,
			},
			"end"
		)
			.to(
			this.material.uniforms.twistAmount,
			{
				value: 0,
			},
			"end"
		)
			.to(
			this.mesh.rotation,
			{
				x: this.mesh.rotation.x - THREE.MathUtils.degToRad(180),
			},
			"end"
		);
	}
}


// ******** INITIALIZE THE "TYPE" INSTANCES ************* //

const OPTION_DEFAULTS = {
	position: {
		texture: [-0.945, -0.5, 0],
		mesh: [0, 0, 0],
	},
	scale: [0.012, 0.04, 1],
	shaders: {
		vertex: vertexShader,
		fragment: fragMentShader,
	},
	font: {
		file: 'https://raw.githubusercontent.com/devloop01/twisting-mesh/gh-pages/fonts/ArchivoBlack-Regular.fnt',
		atlas: 'https://raw.githubusercontent.com/devloop01/twisting-mesh/gh-pages/fonts/ArchivoBlack-Regular.png',
	},
};
const options = [
	{
		word: "HOME",
		color: "#ffffff",
		fill: "#0466c8",
		geometry: new THREE.BoxGeometry(2, 1, 1, 64, 64, 64),
		...OPTION_DEFAULTS,
	},
	{
		word: "WORK",
		color: "#0466c8",
		fill: "#ffffff",
		geometry: new THREE.BoxGeometry(1.9995, 1.0015, 0.9995, 64, 64, 64),
		...OPTION_DEFAULTS,
	},
];

options.forEach((option) => {
	let type = new Type();
	type.init(option);
});