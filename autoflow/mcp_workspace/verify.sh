#!/bin/bash
echo "验证Rust版本..."
rustc --version | grep "1.63.0" || echo "警告：Rust版本不是1.63.0"
echo "验证配置文件..."
[ -f config.yaml ] || echo "错误：缺少config.yaml"

echo "编译所有项目..."
cargo build || { echo "错误：编译失败"; exit 1; }

echo "验证完成！" 