<template>
  <el-tabs :type="type" style="margin-top: 1rem;">
    <el-tab-pane :label="labels[0]"><slot name="g1"></slot></el-tab-pane>
    <el-tab-pane :label="labels[1]"><slot name="g2"></slot></el-tab-pane>
    <el-tab-pane :label="labels[2]" v-if="show"><slot name="g3"></slot></el-tab-pane>
  </el-tabs>
</template>

<script setup>
import { ElTabs, ElTabPane } from 'element-plus'
const props = defineProps({
  type: {
    default: 'border-card'
  },
  labels: {
    default: ['第一组', '第二组', '第三组']
  },
  show: {
    default: true
  }
})
</script>