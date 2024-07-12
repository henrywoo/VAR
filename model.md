# Model Arch

```
🌳 VAR<all params:310283520>
├── Linear(word_embed)|weight[1024,32]|bias[1024]
├── Embedding(class_emb)|weight[1001,1024]
├── Embedding(lvl_embed)|weight[10,1024]
├── ModuleList(blocks)
│   ├── AdaLNSelfAttn(0)
│   │   ├── SelfAttention(attn)|scale_mul_1H11[1,16,1,1]|q_bias[1024]|v_bias[1024]
│   │   │   ├── Linear(mat_qkv)|weight[3072,1024]
│   │   │   └── Linear(proj)|weight[1024,1024]|bias[1024]
│   │   ├── FFN(ffn)
│   │   │   ├── Linear(fc1)|weight[4096,1024]|bias[4096]
│   │   │   └── Linear(fc2)|weight[1024,4096]|bias[1024]
│   │   └── Sequential(ada_lin)
│   │       └── Linear(1)|weight[6144,1024]|bias[6144]
│   └── 💠 AdaLNSelfAttn(1-15)<🦜:18888720x15>
│       ┣━━ SelfAttention(attn)|scale_mul_1H11[1,16,1,1]|q_bias[1024]|v_bias[1024]
│       ┃   ┣━━ Linear(mat_qkv)|weight[3072,1024]
│       ┃   ┗━━ Linear(proj)|weight[1024,1024]|bias[1024]
│       ┣━━ FFN(ffn)
│       ┃   ┣━━ Linear(fc1)|weight[4096,1024]|bias[4096]
│       ┃   ┗━━ Linear(fc2)|weight[1024,4096]|bias[1024]
│       ┗━━ Sequential(ada_lin)
│           ┗━━ Linear(1)|weight[6144,1024]|bias[6144]
├── AdaLNBeforeHead(head_nm)
│   └── Sequential(ada_lin)
│       └── Linear(1)|weight[2048,1024]|bias[2048]
└── Linear(head)|weight[4096,1024]|bias[4096]
```

```
🌳 VQVAE<trainable_params:0,all_params:108948355,percentage:0.00000%>
├── Encoder(encoder)
│   ├── Conv2d(conv_in)|weight[160,3,3,3]❄️|bias[160]❄️
│   ├── ModuleList(down)
│   │   ├── 💠 Module(0-1)<🦜:0,1154080x2>
│   │   │   ┣━━ ModuleList(block)
│   │   │   ┃   ┗━━ 💠 ResnetBlock(0-1)<🦜:0,461760x2>
│   │   │   ┃       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,320x2>|weight[160]❄️|bias[160]❄️
│   │   │   ┃       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,230560x2>|weight[160,160,3,3]❄️|bias[160]❄️
│   │   │   ┗━━ Downsample2x(downsample)
│   │   │       ┗━━ Conv2d(conv)|weight[160,160,3,3]🇸 -(2, 2)❄️|bias[160]🇸 -(2, 2)❄️
│   │   ├── Module(2)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[160]❄️|bias[160]❄️
│   │   │   │   │   ├── Conv2d(conv1)|weight[320,160,3,3]❄️|bias[320]❄️
│   │   │   │   │   ├── GroupNorm(norm2)|weight[320]❄️|bias[320]❄️
│   │   │   │   │   ├── Conv2d(conv2)|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[320,160,1,1]❄️|bias[320]❄️
│   │   │   │   └── ResnetBlock(1)
│   │   │   │       ├── 💠 GroupNorm(norm1,norm2)<🦜:0,640x2>|weight[320]❄️|bias[320]❄️
│   │   │   │       └── 💠 Conv2d(conv1,conv2)<🦜:0,921920x2>|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   └── Downsample2x(downsample)
│   │   │       └── Conv2d(conv)|weight[320,320,3,3]🇸 -(2, 2)❄️|bias[320]🇸 -(2, 2)❄️
│   │   ├── Module(3)
│   │   │   ├── ModuleList(block)
│   │   │   │   └── 💠 ResnetBlock(0-1)<🦜:0,1845120x2>
│   │   │   │       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,640x2>|weight[320]❄️|bias[320]❄️
│   │   │   │       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,921920x2>|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   └── Downsample2x(downsample)
│   │   │       └── Conv2d(conv)|weight[320,320,3,3]🇸 -(2, 2)❄️|bias[320]🇸 -(2, 2)❄️
│   │   └── Module(4)
│   │       ├── ModuleList(block)
│   │       │   ├── ResnetBlock(0)
│   │       │   │   ├── GroupNorm(norm1)|weight[320]❄️|bias[320]❄️
│   │       │   │   ├── Conv2d(conv1)|weight[640,320,3,3]❄️|bias[640]❄️
│   │       │   │   ├── GroupNorm(norm2)|weight[640]❄️|bias[640]❄️
│   │       │   │   ├── Conv2d(conv2)|weight[640,640,3,3]❄️|bias[640]❄️
│   │       │   │   └── Conv2d(nin_shortcut)|weight[640,320,1,1]❄️|bias[640]❄️
│   │       │   └── ResnetBlock(1)
│   │       │       ├── 💠 GroupNorm(norm1,norm2)<🦜:0,1280x2>|weight[640]❄️|bias[640]❄️
│   │       │       └── 💠 Conv2d(conv1,conv2)<🦜:0,3687040x2>|weight[640,640,3,3]❄️|bias[640]❄️
│   │       └── ModuleList(attn)
│   │           └── 💠 AttnBlock(0-1)<🦜:0,1642240x2>
│   │               ┣━━ GroupNorm(norm)|weight[640]❄️|bias[640]❄️
│   │               ┣━━ Conv2d(qkv)|weight[1920,640,1,1]❄️|bias[1920]❄️
│   │               ┗━━ Conv2d(proj_out)|weight[640,640,1,1]❄️|bias[640]❄️
│   ├── Module(mid)
│   │   ├── 💠 ResnetBlock(block_1,block_2)<🦜:0,7376640x2>
│   │   │   ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,1280x2>|weight[640]❄️|bias[640]❄️
│   │   │   ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,3687040x2>|weight[640,640,3,3]❄️|bias[640]❄️
│   │   └── AttnBlock(attn_1)
│   │       ├── GroupNorm(norm)|weight[640]❄️|bias[640]❄️
│   │       ├── Conv2d(qkv)|weight[1920,640,1,1]❄️|bias[1920]❄️
│   │       └── Conv2d(proj_out)|weight[640,640,1,1]❄️|bias[640]❄️
│   ├── GroupNorm(norm_out)|weight[640]❄️|bias[640]❄️
│   └── Conv2d(conv_out)|weight[32,640,3,3]❄️|bias[32]❄️
├── Decoder(decoder)
│   ├── Conv2d(conv_in)|weight[640,32,3,3]❄️|bias[640]❄️
│   ├── Module(mid)
│   │   ├── 💠 ResnetBlock(block_1,block_2)<🦜:0,7376640x2>
│   │   │   ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,1280x2>|weight[640]❄️|bias[640]❄️
│   │   │   ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,3687040x2>|weight[640,640,3,3]❄️|bias[640]❄️
│   │   └── AttnBlock(attn_1)
│   │       ├── GroupNorm(norm)|weight[640]❄️|bias[640]❄️
│   │       ├── Conv2d(qkv)|weight[1920,640,1,1]❄️|bias[1920]❄️
│   │       └── Conv2d(proj_out)|weight[640,640,1,1]❄️|bias[640]❄️
│   ├── ModuleList(up)
│   │   ├── Module(0)
│   │   │   └── ModuleList(block)
│   │   │       └── 💠 ResnetBlock(0-2)<🦜:0,461760x3>
│   │   │           ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,320x2>|weight[160]❄️|bias[160]❄️
│   │   │           ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,230560x2>|weight[160,160,3,3]❄️|bias[160]❄️
│   │   ├── Module(1)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[320]❄️|bias[320]❄️
│   │   │   │   │   ├── Conv2d(conv1)|weight[160,320,3,3]❄️|bias[160]❄️
│   │   │   │   │   ├── GroupNorm(norm2)|weight[160]❄️|bias[160]❄️
│   │   │   │   │   ├── Conv2d(conv2)|weight[160,160,3,3]❄️|bias[160]❄️
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[160,320,1,1]❄️|bias[160]❄️
│   │   │   │   └── 💠 ResnetBlock(1-2)<🦜:0,461760x2>
│   │   │   │       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,320x2>|weight[160]❄️|bias[160]❄️
│   │   │   │       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,230560x2>|weight[160,160,3,3]❄️|bias[160]❄️
│   │   │   └── Upsample2x(upsample)
│   │   │       └── Conv2d(conv)|weight[160,160,3,3]❄️|bias[160]❄️
│   │   ├── Module(2)
│   │   │   ├── ModuleList(block)
│   │   │   │   └── 💠 ResnetBlock(0-2)<🦜:0,1845120x3>
│   │   │   │       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,640x2>|weight[320]❄️|bias[320]❄️
│   │   │   │       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,921920x2>|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   └── Upsample2x(upsample)
│   │   │       └── Conv2d(conv)|weight[320,320,3,3]❄️|bias[320]❄️
│   │   ├── Module(3)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[640]❄️|bias[640]❄️
│   │   │   │   │   ├── Conv2d(conv1)|weight[320,640,3,3]❄️|bias[320]❄️
│   │   │   │   │   ├── GroupNorm(norm2)|weight[320]❄️|bias[320]❄️
│   │   │   │   │   ├── Conv2d(conv2)|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[320,640,1,1]❄️|bias[320]❄️
│   │   │   │   └── 💠 ResnetBlock(1-2)<🦜:0,1845120x2>
│   │   │   │       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,640x2>|weight[320]❄️|bias[320]❄️
│   │   │   │       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,921920x2>|weight[320,320,3,3]❄️|bias[320]❄️
│   │   │   └── Upsample2x(upsample)
│   │   │       └── Conv2d(conv)|weight[320,320,3,3]❄️|bias[320]❄️
│   │   └── Module(4)
│   │       ├── ModuleList(block)
│   │       │   └── 💠 ResnetBlock(0-2)<🦜:0,7376640x3>
│   │       │       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:0,1280x2>|weight[640]❄️|bias[640]❄️
│   │       │       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:0,3687040x2>|weight[640,640,3,3]❄️|bias[640]❄️
│   │       ├── ModuleList(attn)
│   │       │   └── 💠 AttnBlock(0-2)<🦜:0,1642240x3>
│   │       │       ┣━━ GroupNorm(norm)|weight[640]❄️|bias[640]❄️
│   │       │       ┣━━ Conv2d(qkv)|weight[1920,640,1,1]❄️|bias[1920]❄️
│   │       │       ┗━━ Conv2d(proj_out)|weight[640,640,1,1]❄️|bias[640]❄️
│   │       └── Upsample2x(upsample)
│   │           └── Conv2d(conv)|weight[640,640,3,3]❄️|bias[640]❄️
│   ├── GroupNorm(norm_out)|weight[160]❄️|bias[160]❄️
│   └── Conv2d(conv_out)|weight[3,160,3,3]❄️|bias[3]❄️
├── VectorQuantizer2(quantize)
│   ├── PhiPartiallyShared(quant_resi)
│   │   └── ModuleList(qresi_ls)
│   │       └── 💠 Phi(0-3)<🦜:0,9248x4>|weight[32,32,3,3]❄️|bias[32]❄️
│   └── Embedding(embedding)|weight[4096,32]❄️
└── 💠 Conv2d(quant_conv,post_quant_conv)<🦜:0,9248x2>|weight[32,32,3,3]❄️|bias[32]❄️
```