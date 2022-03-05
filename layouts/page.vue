<template>
  <div class="page">
    <div class="icon1"></div>
    <div class="icon2"></div>
    <div class="site-boy"></div>
    <div class="content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.page {
  width: 100%;
  height: 100%;
  padding: 2.5rem 3.5rem;
  padding-right: 12rem;
  background: url(/paper-textured-background.jpg);
}
.icon1 {
  opacity: 1;
  position: absolute;
  /* width: 517.74px;
  height: 359.38px; */
  width: 586px;
  height: 506px;
  transform: scale(1);
  opacity: 1;
  background: url(/top-right.svg);
  transform-origin: right top;
  top: -20%;
  right: -10%;
  z-index: 0;
}
.icon2 {
  opacity: 1;
  position: absolute;
  bottom: -33%;
  left: -8%;
  /* width: 607px;
  height: 284px; */
  width: 612px;
  height: 543px;
  background: url(/bottom-left.svg);
  transform: scale(1);
  transform-origin: left bottom;
  z-index: 0;
}
.site-boy {
  width: 216px;
  height: 251px;
  position: absolute;
  right: 0;
  bottom: 0;
  background: url(/sit-boy.svg);
  z-index: 0;
  transform: scale(.9);
  transform-origin: right bottom;
}
.content {
  position: relative;
  z-index: 1;
}
</style>