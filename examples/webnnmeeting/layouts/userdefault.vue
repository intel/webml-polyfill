<template>
  <section class="hero is-fullheight home">
    <Nav />
    <div class="hero-body">
      <div class="container has-text-centered">
        <nuxt />
      </div>
    </div>
    <Footer />
    <div id="shadow"></div>
  </section>
</template>

<script>
import Nav from '~/components/Nav.vue'
import Footer from '~/components/Footer.vue'

export default {
  head() {
    return {
      script: [
        { src: '../js/socket.io.js', defer: true },
        { src: '../js/adapter-7.0.0.js', defer: true }
      ]
    }
  },
  components: {
    Nav,
    Footer
  }
}
</script>
<style scope>
.leave:focus,
.leave:hover,
.leave.is-active {
  background-color: rgba(0, 0, 0, 0.1) !important;
}

.navbar-menu.is-active {
  background-color: rgba(0, 0, 0, 0.2);
}

.navbar-dropdown .navbar-item:hover,
.navbar-link:hover {
  background-color: rgba(0, 0, 0, 0.4);
}

@media screen and (min-width: 1024px) {
  .navbar-dropdown {
    padding: 0;
  }

  .navbar-item:focus,
  .navbar-item:hover,
  .navbar-item.is-active {
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-item.has-dropdown:focus .navbar-link,
  .navbar-item.has-dropdown:hover .navbar-link,
  .navbar-item.has-dropdown.is-active .navbar-link {
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-dropdown a.navbar-item {
    padding: 0.375rem 1rem;
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-dropdown a.navbar-item:hover {
    padding: 0.375rem 1rem;
    background-color: rgba(0, 0, 0, 0.2);
  }

  .navbar-dropdown {
    background-color: transparent;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
    border-top: 1px solid rgba(204, 255, 144, 1);
    box-shadow: 0 8px 8px rgba(10, 10, 10, 0.1);
    display: none;
    font-size: 0.875rem;
    left: 0;
    min-width: 100%;
    position: absolute;
    top: 100%;
    z-index: 20;
  }
}

@media (max-width: 768px) {
  .navbar-menu {
    background-color: rgba(0, 0, 0, 0.2);
  }

  .logo {
    background-color: transparent;
  }

  a.navbar-item:focus,
  a.navbar-item:focus-within,
  a.navbar-item:hover,
  a.navbar-item.is-active,
  .navbar-link:focus,
  .navbar-link:focus-within,
  .navbar-link:hover,
  .navbar-link.is-active {
    background-color: rgba(0, 0, 0, 0.4);
  }

  .navbar-dropdown {
    padding: 0;
  }
}
</style>
