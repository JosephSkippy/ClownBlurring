document.addEventListener('DOMContentLoaded', function() {
    const toggleSwitch = document.getElementById('toggleSwitch');
  
    // Load the saved state from Chrome storage
    chrome.storage.sync.get(['backgroundEnabled'], function(result) {
      toggleSwitch.checked = result.backgroundEnabled || false;
    });
  
    // Add event listener to the toggle switch
    toggleSwitch.addEventListener('change', function() {
      const isEnabled = toggleSwitch.checked;
      chrome.storage.sync.set({ backgroundEnabled: isEnabled }, function() {
        console.log('Background script enabled:', isEnabled);
      });
  
      // Optionally, send a message to the background script to start/stop its functionality
      chrome.runtime.sendMessage({ msg: 'toggle_background', enabled: isEnabled });
    });
  });